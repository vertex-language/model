// Package gguf reads GGUF, the self-contained weight format of llama.cpp:
// metadata (architecture, hyperparameters, the tokenizer, the chat
// template) and tensors, quantized or not, in one file. The file is
// mapped, not read: a tensor's bytes are used where they lie.
//
// The rules are llama.cpp's (ggml/src/gguf.cpp), so a file it refuses
// this refuses. One more is refused here: a tensor that runs past the end
// of the file, which llama.cpp's header reader lets through and its loader
// finds later. A mapped tensor is read in place, so that is checked first.
package gguf

import "fs"
import "fs/mmap"

/// The magic, and the versions read: v1 had 32-bit counts and is gone.
let magic: uint32 = 0x46554747 // "GGUF", little-endian
let defaultAlignment = 32
let maxDims = 4

/// TensorInfo is where a tensor is and what it holds.
public struct TensorInfo {
    /// Name is the tensor's name in the file: "blk.0.attn_q.weight".
    public let Name: string
    /// Shape is ggml's order, innermost first: Shape[0] is the length of a
    /// row, the elements that lie next to each other.
    public let Shape: [int]
    public let Type: TensorType
    /// Offset is where the tensor's bytes begin, from the file's start.
    public let Offset: int
    /// Size is how many bytes the tensor takes.
    public let Size: int

    /// Count is how many elements the tensor holds.
    public var Count: int {
        var n = 1
        for d in Shape { n *= d }
        return n
    }
}

/// File is an open GGUF file: its metadata, its tensors, and the mapping
/// their bytes are read from, held as long as the File is.
public final class File {
    public let Version: int
    /// Keys are the metadata keys, in the order the file has them.
    public let Keys: [string]
    /// Tensors are the file's tensors, in the order it has them.
    public let Tensors: [TensorInfo]
    /// Alignment is what every tensor's offset is a multiple of.
    public let Alignment: int
    /// DataOffset is where the tensor data begins.
    public let DataOffset: int
    let _values: [string: Value]
    let _index: [string: int]
    let _map: mmap.Mapping

    init(_ version: int, _ keys: [string], _ values: [string: Value], _ tensors: [TensorInfo],
         _ alignment: int, _ dataOffset: int, _ map: mmap.Mapping) {
        self.Version = version
        self.Keys = keys
        self._values = values
        self.Tensors = tensors
        self.Alignment = alignment
        self.DataOffset = dataOffset
        self._map = map
        var index: [string: int] = [:]
        for i in 0..<tensors.count {
            index[tensors[i].Name] = i
        }
        self._index = index
    }

    /// Value is the metadata value under key, or nil.
    public func Value(_ key: string) -> Value? {
        return _values[key]
    }

    /// Text is the string under key, or nil where there is none.
    public func Text(_ key: string) -> string? {
        return _values[key]?.AsText
    }

    /// Integer is the integer under key, of any width, or nil.
    public func Integer(_ key: string) -> int? {
        return _values[key]?.AsInteger
    }

    /// Number is the number under key, float or integer, or nil.
    public func Number(_ key: string) -> float64? {
        return _values[key]?.AsNumber
    }

    /// Flag is the bool under key, or nil.
    public func Flag(_ key: string) -> bool? {
        return _values[key]?.AsFlag
    }

    /// Texts is the array of strings under key, or nil.
    public func Texts(_ key: string) -> [string]? {
        guard case .Array(let t, let xs)? = _values[key], t == .Text else { return nil }
        return xs.map { $0.AsText! }
    }

    /// Integers is the array of integers under key, or nil.
    public func Integers(_ key: string) -> [int]? {
        guard case .Array(_, let xs)? = _values[key] else { return nil }
        var out: [int] = []
        for x in xs {
            guard let n = x.AsInteger else { return nil }
            out.append(n)
        }
        return out
    }

    /// Numbers is the array of numbers under key, or nil.
    public func Numbers(_ key: string) -> [float64]? {
        guard case .Array(_, let xs)? = _values[key] else { return nil }
        var out: [float64] = []
        for x in xs {
            guard let n = x.AsNumber else { return nil }
            out.append(n)
        }
        return out
    }

    /// Architecture is general.architecture: "llama", "qwen3", "gemma3".
    public var Architecture: string? {
        return Text("general.architecture")
    }

    /// Tensor is the tensor named name, or nil.
    public func Tensor(_ name: string) -> TensorInfo? {
        guard let i = _index[name] else { return nil }
        return Tensors[i]
    }

    /// Bytes is where a tensor's bytes lie, in the mapping: valid as long
    /// as this File is.
    public func Bytes(_ t: TensorInfo) -> UnsafePointer<uint8> {
        return _map.Bytes! + t.Offset
    }

    /// Copy is a tensor's bytes, copied out.
    public func Copy(_ t: TensorInfo) -> [uint8] {
        return _map.Copy(from: t.Offset, count: t.Size)
    }
}

/// Open maps the GGUF file at path and reads its header. Tensor bytes are
/// not read until they are used.
public func Open(_ path: fs.Path) throws -> File {
    let m = try mmap.Map(path)
    return try parse(m)
}

func parse(_ m: mmap.Mapping) throws -> File {
    var r = Cursor(m.Bytes, m.Count)
    if try r.U32("the magic") != magic {
        throw FormatError.malformed("not a GGUF file: the magic is not 'GGUF'")
    }
    let version = try r.U32("the version")
    if version == 1 {
        throw FormatError.malformed("GGUF v1 is no longer supported")
    }
    if version & 0xFFFF == 0 {
        throw FormatError.malformed("version \(version): is the file of the other endianness?")
    }
    if version > 3 {
        throw FormatError.malformed("version \(version) is newer than the 3 this reads")
    }
    // A tensor's info is at least 24 bytes, a key-value pair's 13.
    let nTensors = try r.Length(24, "the tensor count")
    let nKV = try r.Length(13, "the metadata count")

    var keys: [string] = []
    var values: [string: Value] = [:]
    for i in 0..<nKV {
        let key = try r.Text("key \(i)")
        if key.isEmpty {
            throw FormatError.malformed("key \(i) is empty")
        }
        if values[key] != nil {
            throw FormatError.malformed("key '\(key)' appears twice")
        }
        values[key] = try readValue(&r, key)
        keys.append(key)
    }

    var alignment = defaultAlignment
    if let a = values["general.alignment"] {
        guard case .U32(let n) = a else {
            throw FormatError.malformed("general.alignment must be a u32")
        }
        if n == 0 || n & (n - 1) != 0 {
            throw FormatError.malformed("alignment \(n) is not a power of 2")
        }
        alignment = int(n)
    }

    var infos: [(string, [int], TensorType, int)] = []
    var names: [string: bool] = [:]
    for i in 0..<nTensors {
        let name = try r.Text("tensor name \(i)")
        if name.utf8.count >= 64 {
            throw FormatError.malformed("tensor name '\(name)' is \(name.utf8.count) bytes, over 63")
        }
        if names[name] != nil {
            throw FormatError.malformed("tensor '\(name)' appears twice")
        }
        names[name] = true
        let nDims = try r.U32("the dimensions of '\(name)'")
        if nDims > uint32(maxDims) {
            throw FormatError.malformed("tensor '\(name)' has \(nDims) dimensions, over \(maxDims)")
        }
        var shape: [int] = []
        var count: int = 1
        for _ in 0..<int(nDims) {
            let d = try r.U64("a dimension of '\(name)'")
            if d > uint64(int.max) {
                throw FormatError.malformed("tensor '\(name)' has a negative dimension")
            }
            let (c, over) = count.multipliedReportingOverflow(by: int(d))
            if over {
                throw FormatError.malformed("tensor '\(name)' has more elements than fit")
            }
            count = c
            shape.append(int(d))
        }
        let raw = try r.U32("the type of '\(name)'")
        guard let type = TensorType(rawValue: raw) else {
            throw FormatError.malformed("tensor '\(name)' has unknown type \(raw)")
        }
        if shape.count > 0 && shape[0] % type.BlockSize != 0 {
            throw FormatError.malformed("tensor '\(name)' of type \(type.Name) has \(shape[0]) elements a row, not a multiple of its block of \(type.BlockSize)")
        }
        let offset = try r.U64("the offset of '\(name)'")
        if offset > uint64(int.max) {
            throw FormatError.malformed("tensor '\(name)' has an offset past any file")
        }
        infos.append((name, shape, type, int(offset)))
    }

    // The data begins at the next multiple of the alignment, and every
    // tensor follows the one before, padded to the alignment.
    let dataOffset = pad(r.Offset, alignment)
    var tensors: [TensorInfo] = []
    var expected = 0
    for (name, shape, type, offset) in infos {
        if offset != expected {
            throw FormatError.malformed("tensor '\(name)' is at offset \(offset), where \(expected) was expected")
        }
        var count = 1
        for d in shape { count *= d }
        let size = count / type.BlockSize * type.TypeSize
        if dataOffset + offset + size > m.Count {
            throw FormatError.malformed("tensor '\(name)' runs past the end of the file")
        }
        tensors.append(TensorInfo(Name: name, Shape: shape, Type: type, Offset: dataOffset + offset, Size: size))
        expected = offset + pad(size, alignment)
    }
    return File(int(version), keys, values, tensors, alignment, dataOffset, m)
}

func pad(_ n: int, _ alignment: int) -> int {
    return (n + alignment - 1) / alignment * alignment
}

func readValue(_ r: inout Cursor, _ key: string) throws -> Value {
    let raw = try r.U32("the type of '\(key)'")
    guard let type = ValueType(rawValue: raw) else {
        throw FormatError.malformed("key '\(key)' has unknown type \(raw)")
    }
    if type != .Array {
        return try readScalar(&r, type, key)
    }
    let rawElem = try r.U32("the element type of '\(key)'")
    guard let elem = ValueType(rawValue: rawElem), elem != .Array else {
        throw FormatError.malformed("key '\(key)' is an array of type \(rawElem), which is not allowed")
    }
    let n = try r.Length(minSize(elem), "the length of '\(key)'")
    var xs: [Value] = []
    xs.reserveCapacity(n)
    for _ in 0..<n {
        xs.append(try readScalar(&r, elem, key))
    }
    return .Array(elem, xs)
}

func minSize(_ t: ValueType) -> int {
    switch t {
    case .U8, .I8, .Bool: return 1
    case .U16, .I16: return 2
    case .U32, .I32, .F32: return 4
    case .U64, .I64, .F64, .Text: return 8
    case .Array: return 12
    }
}

func readScalar(_ r: inout Cursor, _ t: ValueType, _ key: string) throws -> Value {
    switch t {
    case .U8: return .U8(try r.U8(key))
    case .I8: return .I8(int8(bitPattern: try r.U8(key)))
    case .U16: return .U16(try r.U16(key))
    case .I16: return .I16(int16(bitPattern: try r.U16(key)))
    case .U32: return .U32(try r.U32(key))
    case .I32: return .I32(int32(bitPattern: try r.U32(key)))
    case .F32: return .F32(float32(bitPattern: try r.U32(key)))
    case .Bool: return .Bool(try r.U8(key) != 0) // any byte but zero is true, as in llama.cpp
    case .Text: return .Text(try r.Text(key))
    case .U64: return .U64(try r.U64(key))
    case .I64: return .I64(int64(bitPattern: try r.U64(key)))
    case .F64: return .F64(float64(bitPattern: try r.U64(key)))
    case .Array: throw FormatError.malformed("key '\(key)' nests an array")
    }
}
