// Writing GGUF: the reader's rules in reverse, as ggml's gguf_write writes.
package gguf

import "fs"

/// Tensor is a tensor to write: its name, type, shape in ggml's order
/// (innermost first) and its bytes, laid out as the type lays them.
public struct Tensor {
    public var Name: string
    public var Type: TensorType
    public var Shape: [int]
    public var Bytes: [uint8]

    public init(name: string, type: TensorType, shape: [int], bytes: [uint8]) {
        self.Name = name
        self.Type = type
        self.Shape = shape
        self.Bytes = bytes
    }
}

/// Layout is a tensor as the header describes it: its bytes come later.
public struct Layout {
    public var Name: string
    public var Type: TensorType
    public var Shape: [int]

    public init(name: string, type: TensorType, shape: [int]) {
        self.Name = name
        self.Type = type
        self.Shape = shape
    }

    /// Size is how many bytes its data takes.
    public var Size: int {
        var count = 1
        for d in Shape { count *= d }
        return count / Type.BlockSize * Type.TypeSize
    }
}

/// Header is a GGUF v3 file's header -- metadata (in order), then each
/// tensor's name, shape, type and offset -- padded to where the data
/// begins. The tensors' data follows, each padded to 32 bytes.
public func Header(metadata: [(string, Value)], tensors: [Layout]) throws -> [uint8] {
    var w = writer()
    w.u32(magic)
    w.u32(3)
    w.u64(uint64(tensors.count))
    w.u64(uint64(metadata.count))
    for (k, v) in metadata {
        w.text(k)
        w.u32(v.Type.rawValue)
        if case .Array(let elem, let xs) = v {
            if elem == .Array {
                throw FormatError.malformed("\(k): arrays do not nest")
            }
            w.u32(elem.rawValue)
            w.u64(uint64(xs.count))
            for x in xs {
                if x.Type != elem {
                    throw FormatError.malformed("\(k): an array of \(elem) holds a \(x.Type)")
                }
                w.scalar(x)
            }
        } else {
            w.scalar(v)
        }
    }
    var offset = 0
    for t in tensors {
        if t.Shape.isEmpty || t.Shape.count > maxDims {
            throw FormatError.malformed("\(t.Name): \(t.Shape.count) dimensions")
        }
        if t.Shape[0] % t.Type.BlockSize != 0 {
            throw FormatError.malformed("\(t.Name): rows of \(t.Shape[0]) are not whole blocks of \(t.Type.Name)")
        }
        w.text(t.Name)
        w.u32(uint32(t.Shape.count))
        for d in t.Shape {
            w.u64(uint64(d))
        }
        w.u32(t.Type.rawValue)
        w.u64(uint64(offset))
        offset += pad(t.Size, defaultAlignment)
    }
    w.align(defaultAlignment)
    return w.out
}

/// Encode is a GGUF v3 file of metadata and tensors, in memory.
public func Encode(metadata: [(string, Value)], tensors: [Tensor]) throws -> [uint8] {
    let layouts = tensors.map { Layout(name: $0.Name, type: $0.Type, shape: $0.Shape) }
    var out = try Header(metadata: metadata, tensors: layouts)
    for i in 0..<tensors.count {
        if tensors[i].Bytes.count != layouts[i].Size {
            throw FormatError.malformed("\(tensors[i].Name): \(tensors[i].Bytes.count) bytes for \(tensors[i].Shape) of \(tensors[i].Type.Name)")
        }
        out.append(contentsOf: tensors[i].Bytes)
        while out.count % defaultAlignment != 0 { out.append(0) }
    }
    return out
}

/// Save writes a GGUF file.
public func Save(metadata: [(string, Value)], tensors: [Tensor], to path: fs.Path) throws {
    try fs.WriteFile(path, try Encode(metadata: metadata, tensors: tensors), atomic: true)
}

/// Save writes a GGUF file a tensor at a time: bytes(i) is tensor i's data,
/// asked for as it is written, so a model larger than memory converts.
public func Save(metadata: [(string, Value)], tensors: [Layout], to path: fs.Path, bytes: (int) throws -> [uint8]) throws {
    let partial = fs.Path(path.String() + ".partial")
    let f = try fs.Create(partial)
    do {
        var at = 0
        let head = try Header(metadata: metadata, tensors: tensors)
        try f.Write(head)
        at += head.count
        for i in 0..<tensors.count {
            let b = try bytes(i)
            if b.count != tensors[i].Size {
                throw FormatError.malformed("\(tensors[i].Name): \(b.count) bytes for \(tensors[i].Shape) of \(tensors[i].Type.Name)")
            }
            try f.Write(b)
            at += b.count
            let padding = pad(at, defaultAlignment) - at
            if padding > 0 {
                try f.Write([uint8](repeating: 0, count: padding))
                at += padding
            }
        }
        try f.Close()
    } catch {
        try? f.Close()
        try? fs.Remove(partial)
        throw error
    }
    try fs.Rename(partial, path)
}

struct writer {
    var out: [uint8] = []

    mutating func u8(_ v: uint8) { out.append(v) }
    mutating func u16(_ v: uint16) {
        out.append(uint8(truncatingIfNeeded: v))
        out.append(uint8(truncatingIfNeeded: v >> 8))
    }
    mutating func u32(_ v: uint32) {
        for i in 0..<4 { out.append(uint8(truncatingIfNeeded: v >> uint32(8 * i))) }
    }
    mutating func u64(_ v: uint64) {
        for i in 0..<8 { out.append(uint8(truncatingIfNeeded: v >> uint64(8 * i))) }
    }
    mutating func text(_ s: string) {
        let b = [uint8](s.utf8)
        u64(uint64(b.count))
        out.append(contentsOf: b)
    }
    mutating func align(_ a: int) {
        while out.count % a != 0 { out.append(0) }
    }
    mutating func scalar(_ v: Value) {
        switch v {
        case .U8(let x): u8(x)
        case .I8(let x): u8(uint8(bitPattern: x))
        case .U16(let x): u16(x)
        case .I16(let x): u16(uint16(bitPattern: x))
        case .U32(let x): u32(x)
        case .I32(let x): u32(uint32(bitPattern: x))
        case .F32(let x): u32(x.bitPattern)
        case .Bool(let x): u8(x ? 1 : 0)
        case .Text(let x): text(x)
        case .U64(let x): u64(x)
        case .I64(let x): u64(uint64(bitPattern: x))
        case .F64(let x): u64(x.bitPattern)
        case .Array: break
        }
    }
}
