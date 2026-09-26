// Package safetensors reads safetensors, the Hugging Face Hub's canonical
// weight format: an 8-byte little-endian header length, a JSON header
// naming each tensor's dtype, shape and byte range, then the tensors' raw
// little-endian bytes. The file is mapped, not read: a tensor's bytes are
// used where they lie.
//
// The rules are the Rust safetensors crate's (safetensors/src/tensor.rs),
// so a file it refuses this refuses: the header must fit the file, every
// tensor's byte range must match its shape and dtype, and the ranges must
// tile the data exactly -- no gaps, no overlaps, nothing after.
//
// The package knows nothing about models: names are what the file says,
// and sharding (model.safetensors.index.json) is the repository's
// convention, which model/checkpoint reads.
package safetensors

import (
    "encoding/json"
    "fs"
    "fs/mmap"
)

// A header this big is refused before it is parsed, as the crate does.
let maxHeader = 100_000_000

/// DType is a tensor's element type, as the format names it.
public enum DType: Equatable {
    case BOOL
    case U8
    case I8
    case F8_E5M2
    case F8_E4M3
    case F8_E8M0
    case I16
    case U16
    case F16
    case BF16
    case I32
    case U32
    case F32
    case F64
    case I64
    case U64
    /// Sub-byte floats, packed: 4 and 6 bits an element.
    case F4
    case F6_E2M3
    case F6_E3M2

    /// Name is the format's spelling: "BF16".
    public var Name: string {
        switch self {
        case .BOOL: return "BOOL"
        case .U8: return "U8"
        case .I8: return "I8"
        case .F8_E5M2: return "F8_E5M2"
        case .F8_E4M3: return "F8_E4M3"
        case .F8_E8M0: return "F8_E8M0"
        case .I16: return "I16"
        case .U16: return "U16"
        case .F16: return "F16"
        case .BF16: return "BF16"
        case .I32: return "I32"
        case .U32: return "U32"
        case .F32: return "F32"
        case .F64: return "F64"
        case .I64: return "I64"
        case .U64: return "U64"
        case .F4: return "F4"
        case .F6_E2M3: return "F6_E2M3"
        case .F6_E3M2: return "F6_E3M2"
        }
    }

    /// Bits is how many bits an element takes.
    public var Bits: int {
        switch self {
        case .F4: return 4
        case .F6_E2M3, .F6_E3M2: return 6
        case .BOOL, .U8, .I8, .F8_E5M2, .F8_E4M3, .F8_E8M0: return 8
        case .I16, .U16, .F16, .BF16: return 16
        case .I32, .U32, .F32: return 32
        case .F64, .I64, .U64: return 64
        }
    }

    /// Parse is the dtype a header names, or nil.
    public static func Parse(_ name: string) -> DType? {
        for d in all {
            if d.Name == name {
                return d
            }
        }
        return nil
    }

    static let all: [DType] = [.BOOL, .U8, .I8, .F8_E5M2, .F8_E4M3, .F8_E8M0, .I16, .U16, .F16, .BF16,
                               .I32, .U32, .F32, .F64, .I64, .U64, .F4, .F6_E2M3, .F6_E3M2]
}

/// TensorInfo is where a tensor is and what it holds.
public struct TensorInfo {
    public let Name: string
    /// Shape is outermost first, as PyTorch has it: a Linear's weight is
    /// [out, in], each of its out rows in elements lying together.
    public let Shape: [int]
    public let DType: DType
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

/// FormatError is a file this refuses, and why.
public enum FormatError: Error, CustomStringConvertible {
    case malformed(string)

    public var Message: string {
        switch self {
        case .malformed(let why): return "safetensors: " + why
        }
    }

    public var description: string { return Message }
}

/// File is an open safetensors file: its tensors, its metadata, and the
/// mapping their bytes are read from, held as long as the File is.
public final class File {
    /// Tensors are the file's tensors, in the order of their bytes.
    public let Tensors: [TensorInfo]
    /// Metadata is the header's __metadata__: free-form strings, often
    /// just {"format": "pt"}.
    public let Metadata: [string: string]
    /// DataOffset is where the tensor data begins: 8 + the header's length.
    public let DataOffset: int
    let _index: [string: int]
    let _map: mmap.Mapping

    init(_ tensors: [TensorInfo], _ metadata: [string: string], _ dataOffset: int, _ map: mmap.Mapping) {
        self.Tensors = tensors
        self.Metadata = metadata
        self.DataOffset = dataOffset
        self._map = map
        var index: [string: int] = [:]
        for i in 0..<tensors.count {
            index[tensors[i].Name] = i
        }
        self._index = index
    }

    /// Tensor is the tensor named name, or nil.
    public func Tensor(_ name: string) -> TensorInfo? {
        guard let i = _index[name] else { return nil }
        return Tensors[i]
    }

    /// Mapping is the file's bytes in memory, which tensors are read from
    /// in place: a device can take all of it as one buffer (gpu.Device.Wrap).
    public var Mapping: mmap.Mapping { return _map }

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

/// Open maps the safetensors file at path and reads its header. Tensor
/// bytes are not read until they are used.
public func Open(_ path: fs.Path) throws -> File {
    let m = try mmap.Map(path)
    return try parse(m)
}

func parse(_ m: mmap.Mapping) throws -> File {
    if m.Count < 8 {
        throw FormatError.malformed("\(m.Count) bytes: too short for the header length")
    }
    let lenBytes = m.Copy(from: 0, count: 8)
    var n: uint64 = 0
    var i = 7
    while i >= 0 {
        n = (n << 8) | uint64(lenBytes[i])
        i -= 1
    }
    if n > uint64(maxHeader) {
        throw FormatError.malformed("a header of \(n) bytes is past the limit of \(maxHeader)")
    }
    let headerLen = int(n)
    if 8 + headerLen > m.Count {
        throw FormatError.malformed("a header of \(headerLen) bytes runs past the file's \(m.Count)")
    }
    let header = m.Copy(from: 8, count: headerLen)
    if header.isEmpty || header[0] != 0x7B {
        throw FormatError.malformed("the header does not start with '{'")
    }
    var doc: json.Value
    do {
        doc = try json.Parse(bytes: header)
    } catch {
        throw FormatError.malformed("the header is not JSON: \(error)")
    }
    guard let obj = doc.Object else {
        throw FormatError.malformed("the header is not a JSON object")
    }
    let dataOffset = 8 + headerLen
    let dataLen = m.Count - dataOffset

    var metadata: [string: string] = [:]
    var tensors: [TensorInfo] = []
    for (name, v) in obj.Members {
        if name == "__metadata__" {
            guard let meta = v.Object else {
                throw FormatError.malformed("__metadata__ is not an object")
            }
            for (k, mv) in meta.Members {
                guard let text = mv.String else {
                    throw FormatError.malformed("__metadata__ \(k) is not a string")
                }
                metadata[k] = text
            }
            continue
        }
        guard let dname = v["dtype"]?.String, let dt = DType.Parse(dname) else {
            throw FormatError.malformed("\(name): no dtype, or one this does not know")
        }
        guard let dims = v["shape"]?.Array else {
            throw FormatError.malformed("\(name): no shape")
        }
        var shape: [int] = []
        var count = 1
        for d in dims {
            guard let x = d.Int, x >= 0, x < (1 << 48) else {
                throw FormatError.malformed("\(name): shape \(json.Encode(v["shape"]!)) is not of non-negative integers")
            }
            shape.append(int(x))
            if x > 0 && count > (1 << 56) / int(x) {
                throw FormatError.malformed("\(name): shape overflows")
            }
            count *= int(x)
        }
        guard let offs = v["data_offsets"]?.Array, offs.count == 2,
              let begin = offs[0].Int, let end = offs[1].Int, begin >= 0, end >= begin else {
            throw FormatError.malformed("\(name): data_offsets are not [begin, end]")
        }
        let bits = count * dt.Bits
        if bits % 8 != 0 {
            throw FormatError.malformed("\(name): \(count) elements of \(dt.Name) are not whole bytes")
        }
        if int(end - begin) != bits / 8 {
            throw FormatError.malformed("\(name): \(end - begin) bytes for \(shape) of \(dt.Name), which is \(bits / 8)")
        }
        if int(end) > dataLen {
            throw FormatError.malformed("\(name): bytes \(begin)..<\(end) run past the data's \(dataLen)")
        }
        tensors.append(TensorInfo(Name: name, Shape: shape, DType: dt, Offset: dataOffset + int(begin), Size: int(end - begin)))
    }

    // The byte ranges, in order, tile the data: each begins where the one
    // before ended, and the last ends at the file's end.
    tensors.sort { a, b in a.Offset < b.Offset || (a.Offset == b.Offset && a.Size < b.Size) }
    var at = dataOffset
    for t in tensors {
        if t.Offset != at {
            throw FormatError.malformed("\(t.Name): begins at byte \(t.Offset - dataOffset) of the data, not \(at - dataOffset): a gap or an overlap")
        }
        at += t.Size
    }
    if at != m.Count {
        throw FormatError.malformed("the tensors end at byte \(at - dataOffset) of the data, which runs to \(dataLen)")
    }
    return File(tensors, metadata, dataOffset, m)
}

/// Entry is a tensor to write: its name, dtype, shape and bytes.
public struct Entry {
    public var Name: string
    public var DType: DType
    public var Shape: [int]
    public var Bytes: [uint8]

    public init(name: string, dtype: DType, shape: [int], bytes: [uint8]) {
        self.Name = name
        self.DType = dtype
        self.Shape = shape
        self.Bytes = bytes
    }
}

/// Encode is a safetensors file of entries, in their order, and metadata.
/// The header is padded with spaces to a multiple of 8 bytes, as the Rust
/// crate pads it, so every tensor's bytes start as aligned as the file.
public func Encode(_ entries: [Entry], metadata: [string: string] = [:]) throws -> [uint8] {
    var head = json.Object()
    if !metadata.isEmpty {
        var meta = json.Object()
        for k in metadata.keys.sorted() {
            meta[k] = json.Value.string(metadata[k]!)
        }
        head["__metadata__"] = json.Value.object(meta)
    }
    var at = 0
    for e in entries {
        var count = 1
        for d in e.Shape { count *= d }
        if count * e.DType.Bits != e.Bytes.count * 8 {
            throw FormatError.malformed("\(e.Name): \(e.Bytes.count) bytes for \(e.Shape) of \(e.DType.Name)")
        }
        if head[e.Name] != nil {
            throw FormatError.malformed("\(e.Name) twice")
        }
        var t = json.Object()
        t["dtype"] = json.Value.string(e.DType.Name)
        t["shape"] = json.Value.array(e.Shape.map { json.Value.number("\($0)") })
        t["data_offsets"] = json.Value.array([json.Value.number("\(at)"), json.Value.number("\(at + e.Bytes.count)")])
        head[e.Name] = json.Value.object(t)
        at += e.Bytes.count
    }
    var header = [uint8](json.Encode(json.Value.object(head)).utf8)
    while header.count % 8 != 0 {
        header.append(0x20)
    }
    var out: [uint8] = []
    out.reserveCapacity(8 + header.count + at)
    var n = uint64(header.count)
    for _ in 0..<8 {
        out.append(uint8(truncatingIfNeeded: n))
        n >>= 8
    }
    out.append(contentsOf: header)
    for e in entries {
        out.append(contentsOf: e.Bytes)
    }
    return out
}

/// Save writes entries to path as a safetensors file.
public func Save(_ entries: [Entry], to path: fs.Path, metadata: [string: string] = [:]) throws {
    try fs.WriteFile(path, try Encode(entries, metadata: metadata), atomic: true)
}
