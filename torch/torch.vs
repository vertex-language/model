// Package torch reads PyTorch checkpoints -- the zip files torch.save
// writes (.pth, .pt, pytorch_model.bin) -- as tensors by name, the file
// mapped and nothing copied.
//
// A checkpoint is data.pkl, a pickle describing the saved object, beside
// data/0, data/1, …, each tensor storage's raw bytes, stored uncompressed.
// The pickle is read by a restricted unpickler: the opcodes torch.save
// emits, and only the globals that rebuild tensors and state dicts
// (collections.OrderedDict, torch._utils._rebuild_tensor_v2 and
// _rebuild_parameter, and the storage types). Anything else is refused --
// the line PyTorch draws with weights_only=True -- so opening a checkpoint
// never runs code.
//
// Nested dictionaries flatten to dotted names, as a state dict's are:
// {"decoder": {"conv.weight": t}} is "decoder.conv.weight". A file holding
// one tensor (torch.save(t)) has one, named "".
package torch

import (
    "fs"
    "fs/mmap"
)

/// DType is a storage's element type.
public enum DType: Equatable {
    case F64
    case F32
    case F16
    case BF16
    case I64
    case I32
    case I16
    case I8
    case U8
    case Bool

    /// Name is PyTorch's: "float32".
    public var Name: string {
        switch self {
        case .F64: return "float64"
        case .F32: return "float32"
        case .F16: return "float16"
        case .BF16: return "bfloat16"
        case .I64: return "int64"
        case .I32: return "int32"
        case .I16: return "int16"
        case .I8: return "int8"
        case .U8: return "uint8"
        case .Bool: return "bool"
        }
    }

    /// Size is an element's bytes.
    public var Size: int {
        switch self {
        case .F64, .I64: return 8
        case .F32, .I32: return 4
        case .F16, .BF16, .I16: return 2
        case .I8, .U8, .Bool: return 1
        }
    }
}

/// TensorInfo is a tensor in the file: its name, shape (outermost first),
/// element type, and where its bytes lie in the file.
public struct TensorInfo {
    public let Name: string
    public let Shape: [int]
    public let DType: DType
    /// Offset is the tensor's first byte in the file.
    public let Offset: int
    /// Size is the tensor's bytes.
    public let Size: int

    /// Count is the tensor's elements.
    public var Count: int {
        var n = 1
        for d in Shape { n *= d }
        return n
    }
}

/// FormatError is a file that is not a checkpoint this reads, and why.
public enum FormatError: Error, CustomStringConvertible {
    case invalid(string)
    case refused(string)

    public var Message: string {
        switch self {
        case .invalid(let s): return "torch: " + s
        case .refused(let s): return "torch: refused: " + s
        }
    }

    public var description: string { return Message }
}

/// File is an opened checkpoint.
public final class File {
    /// Tensors are the file's tensors, in the order the pickle names them.
    public let Tensors: [TensorInfo]
    let _map: mmap.Mapping
    let _byName: [string: int]

    init(_ map: mmap.Mapping, _ tensors: [TensorInfo]) {
        self._map = map
        self.Tensors = tensors
        var byName: [string: int] = [:]
        for i in 0..<tensors.count {
            byName[tensors[i].Name] = i
        }
        self._byName = byName
    }

    /// Tensor is the named tensor, or nil.
    public func Tensor(_ name: string) -> TensorInfo? {
        guard let i = _byName[name] else { return nil }
        return Tensors[i]
    }

    /// Mapping is the file's bytes.
    public var Mapping: mmap.Mapping { return _map }

    /// Bytes is where a tensor's bytes lie in the mapping.
    public func Bytes(_ t: TensorInfo) -> UnsafePointer<uint8> {
        return _map.Bytes! + t.Offset
    }

    /// Copy is a tensor's bytes, copied out.
    public func Copy(_ t: TensorInfo) -> [uint8] {
        return _map.Copy(from: t.Offset, count: t.Size)
    }
}

/// Open maps a PyTorch checkpoint and reads its tensors' names, shapes and
/// places; no tensor byte is read.
public func Open(_ path: fs.Path) throws -> File {
    let map = try mmap.Map(path)
    guard let base = map.Bytes else {
        throw FormatError.invalid("\(path.String()) is empty")
    }
    let entries = try zipEntries(base, map.Count)
    var pickle = ""
    var prefix = ""
    for (name, _) in entries where name.hasSuffix("/data.pkl") || name == "data.pkl" {
        pickle = name
        prefix = String(name.dropLast("data.pkl".count))
        break
    }
    guard let pkl = entries[pickle] else {
        throw FormatError.invalid("\(path.String()) has no data.pkl: not a torch.save zip")
    }
    if let order = entries[prefix + "byteorder"] {
        let text = string(decoding: map.Copy(from: order.offset, count: order.size), as: UTF8.self)
        if text != "little" {
            throw FormatError.invalid("a \(text)-endian checkpoint")
        }
    }
    var u = Unpickler(base + pkl.offset, pkl.size)
    let root = try u.Load()

    var tensors: [TensorInfo] = []
    func add(_ name: string, _ v: Value) throws {
        switch v {
        case .tensor(let t):
            guard let e = entries[prefix + "data/" + t.key] else {
                throw FormatError.invalid("\(name)'s storage data/\(t.key) is not in the file")
            }
            var count = 1
            for d in t.shape { count *= d }
            // Only a contiguous, row-major view of its storage is a tensor
            // whose bytes lie in one run.
            var expect = 1
            var i = t.shape.count - 1
            while i >= 0 {
                if t.shape[i] != 1 && t.stride[i] != expect {
                    throw FormatError.invalid("\(name) is not contiguous (strides \(t.stride) for shape \(t.shape))")
                }
                expect *= t.shape[i]
                i -= 1
            }
            let start = t.offset * t.dtype.Size
            let size = count * t.dtype.Size
            if start + size > e.size {
                throw FormatError.invalid("\(name) runs past its storage")
            }
            tensors.append(TensorInfo(Name: name, Shape: t.shape, DType: t.dtype, Offset: e.offset + start, Size: size))
        case .dict(let items):
            for (k, x) in items {
                guard case .string(let key) = k else {
                    throw FormatError.invalid("a dictionary key that is not a string, under '\(name)'")
                }
                try add(name.isEmpty ? key : name + "." + key, x)
            }
        default:
            break // metadata beside the tensors: a version number, a flag
        }
    }
    try add("", root)
    return File(map, tensors)
}

// ---- the zip directory ----

struct Entry {
    var offset: int // the data's first byte in the file
    var size: int
}

// copied is n bytes from p.
func copied(_ p: UnsafePointer<uint8>, _ n: int) -> [uint8] {
    var out = [uint8](repeating: 0, count: n)
    var i = 0
    while i < n {
        out[i] = p[i]
        i += 1
    }
    return out
}

func u16(_ p: UnsafePointer<uint8>, _ i: int) -> int {
    return int(p[i]) | int(p[i + 1]) << 8
}

func u32(_ p: UnsafePointer<uint8>, _ i: int) -> int {
    return int(p[i]) | int(p[i + 1]) << 8 | int(p[i + 2]) << 16 | int(p[i + 3]) << 24
}

func u64(_ p: UnsafePointer<uint8>, _ i: int) -> int {
    return u32(p, i) | u32(p, i + 4) << 32
}

// zipEntries reads the central directory over the mapping, zip64
// included: every stored entry's name and the place of its data. A
// deflated entry is refused, since its bytes could not be read in place.
func zipEntries(_ p: UnsafePointer<uint8>, _ n: int) throws -> [string: Entry] {
    var eocd = -1
    var i = n - 22
    let stop = max(0, n - 22 - 65535)
    while i >= stop {
        if p[i] == 0x50 && p[i + 1] == 0x4b && p[i + 2] == 0x05 && p[i + 3] == 0x06 {
            eocd = i
            break
        }
        i -= 1
    }
    if eocd < 0 {
        throw FormatError.invalid("not a zip file: no end of central directory")
    }
    var count = u16(p, eocd + 10)
    var dir = u32(p, eocd + 16)
    // zip64: the end-of-directory locator, just before, points at the
    // zip64 record, which has the real counts.
    if (count == 0xFFFF || dir == 0xFFFFFFFF) && eocd >= 20 && u32(p, eocd - 20) == 0x07064b50 {
        let rec = u64(p, eocd - 20 + 8)
        if rec + 56 > n || u32(p, rec) != 0x06064b50 {
            throw FormatError.invalid("a broken zip64 end record")
        }
        count = u64(p, rec + 32)
        dir = u64(p, rec + 48)
    }
    var out: [string: Entry] = [:]
    var at = dir
    for _ in 0..<count {
        if at + 46 > n || u32(p, at) != 0x02014b50 {
            throw FormatError.invalid("a broken central directory")
        }
        let method = u16(p, at + 10)
        var size = u32(p, at + 20)
        let nameLen = u16(p, at + 28)
        let extraLen = u16(p, at + 30)
        let commentLen = u16(p, at + 32)
        var local = u32(p, at + 42)
        let name = string(decoding: copied(p + at + 46, nameLen), as: UTF8.self)
        // A zip64 extra field holds the sizes and offset the 32-bit fields
        // could not, in that order, for those that are 0xFFFFFFFF.
        var e = at + 46 + nameLen
        let end = e + extraLen
        while e + 4 <= end {
            let id = u16(p, e)
            let len = u16(p, e + 2)
            if id == 1 {
                var f = e + 4
                if u32(p, at + 24) == 0xFFFFFFFF { f += 8 } // uncompressed size, before the compressed
                if size == 0xFFFFFFFF { size = u64(p, f); f += 8 }
                if local == 0xFFFFFFFF { local = u64(p, f) }
            }
            e += 4 + len
        }
        if method != 0 {
            throw FormatError.invalid("\(name) is compressed; a checkpoint's entries are stored")
        }
        if local + 30 > n || u32(p, local) != 0x04034b50 {
            throw FormatError.invalid("\(name)'s local header is broken")
        }
        let data = local + 30 + u16(p, local + 26) + u16(p, local + 28)
        if data + size > n {
            throw FormatError.invalid("\(name) runs past the end of the file")
        }
        out[name] = Entry(offset: data, size: size)
        at += 46 + nameLen + extraLen + commentLen
    }
    return out
}

// ---- the restricted unpickler ----

struct TensorRef {
    var key: string
    var dtype: DType
    var offset: int
    var shape: [int]
    var stride: [int]
}

indirect enum Value {
    case none
    case bool(bool)
    case int(int)
    case float(float64)
    case string(string)
    case bytes([uint8])
    case tuple([Value])
    case list([Value])
    case dict([(Value, Value)])
    case global(string, string)
    case storage(string, DType)
    case tensor(TensorRef)
    case mark
}

// The globals a checkpoint may name, and what each storage type holds.
let storageTypes: [string: DType] = [
    "DoubleStorage": .F64, "FloatStorage": .F32, "HalfStorage": .F16, "BFloat16Storage": .BF16,
    "LongStorage": .I64, "IntStorage": .I32, "ShortStorage": .I16, "CharStorage": .I8,
    "ByteStorage": .U8, "BoolStorage": .Bool,
]

func allowed(_ module: string, _ name: string) -> bool {
    switch module + "." + name {
    case "collections.OrderedDict", "torch._utils._rebuild_tensor_v2", "torch._utils._rebuild_parameter":
        return true
    default:
        return module == "torch" && storageTypes[name] != nil
    }
}

struct Unpickler {
    let p: UnsafePointer<uint8>
    let n: int
    var at: int = 0
    var stack: [Value] = []
    var marks: [int] = []
    var memo: [int: Value] = [:]

    init(_ p: UnsafePointer<uint8>, _ n: int) {
        self.p = p
        self.n = n
    }

    mutating func byte() throws -> int {
        if at >= n { throw FormatError.invalid("the pickle ends early") }
        let b = int(p[at])
        at += 1
        return b
    }

    mutating func take(_ k: int) throws -> UnsafePointer<uint8> {
        if k < 0 || at + k > n { throw FormatError.invalid("the pickle ends early") }
        let q = p + at
        at += k
        return q
    }

    mutating func le(_ k: int) throws -> int {
        let q = try take(k)
        var v = 0
        var i = k - 1
        while i >= 0 {
            v = v << 8 | int(q[i])
            i -= 1
        }
        return v
    }

    mutating func text(_ k: int) throws -> string {
        let q = try take(k)
        return string(decoding: copied(q, k), as: UTF8.self)
    }

    mutating func line() throws -> string {
        var s: [uint8] = []
        while true {
            let b = try byte()
            if b == 0x0a { break }
            s.append(uint8(b))
        }
        return string(decoding: s, as: UTF8.self)
    }

    mutating func pop() throws -> Value {
        guard let v = stack.popLast() else { throw FormatError.invalid("the pickle's stack is empty") }
        return v
    }

    // popMark is everything above the last mark.
    mutating func popMark() throws -> [Value] {
        guard let m = marks.popLast() else { throw FormatError.invalid("no mark") }
        let items = Array(stack[m..<stack.count])
        while stack.count > m {
            stack.removeLast()
        }
        return items
    }

    mutating func Load() throws -> Value {
        while true {
            let op = try byte()
            switch op {
            case 0x80: _ = try byte() // PROTO
            case 0x95: _ = try take(8) // FRAME
            case 0x2e: return try pop() // STOP
            case 0x28: marks.append(stack.count) // MARK
            case 0x4e: stack.append(.none) // NONE
            case 0x88: stack.append(.bool(true)) // NEWTRUE
            case 0x89: stack.append(.bool(false)) // NEWFALSE
            case 0x4a: // BININT: a signed 32-bit int
                var v = try le(4)
                if v >= 1 << 31 { v -= 1 << 32 }
                stack.append(.int(v))
            case 0x4b: stack.append(.int(try le(1))) // BININT1
            case 0x4d: stack.append(.int(try le(2))) // BININT2
            case 0x8a: // LONG1: a little-endian two's complement of n bytes
                let k = try byte()
                if k > 8 { throw FormatError.invalid("an integer of \(k) bytes") }
                var v = k == 0 ? 0 : try le(k)
                if k > 0 && k < 8 && v >= 1 << (8 * k - 1) { v -= 1 << (8 * k) }
                stack.append(.int(v))
            case 0x47: // BINFLOAT: a big-endian float64
                let q = try take(8)
                var bits: uint64 = 0
                for i in 0..<8 { bits = bits << 8 | uint64(q[i]) }
                stack.append(.float(float64(bitPattern: bits)))
            case 0x58: stack.append(.string(try text(try le(4)))) // BINUNICODE
            case 0x8c: stack.append(.string(try text(try le(1)))) // SHORT_BINUNICODE
            case 0x8d: stack.append(.string(try text(try le(8)))) // BINUNICODE8
            case 0x43: // SHORT_BINBYTES
                let k = try le(1)
                stack.append(.bytes(copied(try take(k), k)))
            case 0x42: // BINBYTES
                let k = try le(4)
                stack.append(.bytes(copied(try take(k), k)))
            case 0x29: stack.append(.tuple([])) // EMPTY_TUPLE
            case 0x5d: stack.append(.list([])) // EMPTY_LIST
            case 0x7d: stack.append(.dict([])) // EMPTY_DICT
            case 0x74: stack.append(.tuple(try popMark())) // TUPLE
            case 0x85: stack.append(.tuple([try pop()])) // TUPLE1
            case 0x86: // TUPLE2
                let b = try pop()
                let a = try pop()
                stack.append(.tuple([a, b]))
            case 0x87: // TUPLE3
                let c = try pop()
                let b = try pop()
                let a = try pop()
                stack.append(.tuple([a, b, c]))
            case 0x6c: stack.append(.list(try popMark())) // LIST
            case 0x61: // APPEND
                let v = try pop()
                guard case .list(var xs) = try pop() else { throw FormatError.invalid("APPEND to a non-list") }
                xs.append(v)
                stack.append(.list(xs))
            case 0x65: // APPENDS
                let vs = try popMark()
                guard case .list(var xs) = try pop() else { throw FormatError.invalid("APPENDS to a non-list") }
                xs += vs
                stack.append(.list(xs))
            case 0x64: // DICT
                let kv = try popMark()
                var items: [(Value, Value)] = []
                var i = 0
                while i + 1 < kv.count {
                    items.append((kv[i], kv[i + 1]))
                    i += 2
                }
                stack.append(.dict(items))
            case 0x73: // SETITEM
                let v = try pop()
                let k = try pop()
                guard case .dict(var items) = try pop() else { throw FormatError.invalid("SETITEM on a non-dict") }
                items.append((k, v))
                stack.append(.dict(items))
            case 0x75: // SETITEMS
                let kv = try popMark()
                guard case .dict(var items) = try pop() else { throw FormatError.invalid("SETITEMS on a non-dict") }
                var i = 0
                while i + 1 < kv.count {
                    items.append((kv[i], kv[i + 1]))
                    i += 2
                }
                stack.append(.dict(items))
            case 0x71: memo[try le(1)] = stack.last ?? .none // BINPUT
            case 0x72: memo[try le(4)] = stack.last ?? .none // LONG_BINPUT
            case 0x94: memo[memo.count] = stack.last ?? .none // MEMOIZE
            case 0x68: // BINGET
                guard let v = memo[try le(1)] else { throw FormatError.invalid("BINGET of nothing") }
                stack.append(v)
            case 0x6a: // LONG_BINGET
                guard let v = memo[try le(4)] else { throw FormatError.invalid("LONG_BINGET of nothing") }
                stack.append(v)
            case 0x63: // GLOBAL: "module\nname\n"
                let module = try line()
                let name = try line()
                if !allowed(module, name) {
                    throw FormatError.refused("the global \(module).\(name)")
                }
                stack.append(.global(module, name))
            case 0x93: // STACK_GLOBAL
                guard case .string(let name) = try pop(), case .string(let module) = try pop() else {
                    throw FormatError.invalid("STACK_GLOBAL of non-strings")
                }
                if !allowed(module, name) {
                    throw FormatError.refused("the global \(module).\(name)")
                }
                stack.append(.global(module, name))
            case 0x51: // BINPERSID: ("storage", type, key, location, count)
                guard case .tuple(let pid) = try pop(), pid.count >= 3,
                      case .string(let kind) = pid[0], kind == "storage",
                      case .global(_, let type) = pid[1], let dt = storageTypes[type] else {
                    throw FormatError.invalid("a persistent id that is not a storage")
                }
                var key = ""
                switch pid[2] {
                case .string(let s): key = s
                case .int(let k): key = "\(k)"
                default: throw FormatError.invalid("a storage key that is not a string")
                }
                stack.append(.storage(key, dt))
            case 0x52: // REDUCE: callable(args)
                guard case .tuple(let args) = try pop(), case .global(let module, let name) = try pop() else {
                    throw FormatError.invalid("REDUCE of something not a global")
                }
                stack.append(try call(module, name, args))
            case 0x62: // BUILD: sets an object's state; a state dict's _metadata
                _ = try pop()
            default:
                throw FormatError.refused("pickle opcode 0x\(string(op, radix: 16))")
            }
        }
    }

    func call(_ module: string, _ name: string, _ args: [Value]) throws -> Value {
        switch name {
        case "OrderedDict":
            return .dict([])
        case "_rebuild_tensor_v2":
            // (storage, storage_offset, size, stride, requires_grad, hooks[, metadata])
            guard args.count >= 4, case .storage(let key, let dt) = args[0], case .int(let offset) = args[1],
                  case .tuple(let size) = args[2], case .tuple(let stride) = args[3] else {
                throw FormatError.invalid("_rebuild_tensor_v2 with arguments it does not take")
            }
            return .tensor(TensorRef(key: key, dtype: dt, offset: offset, shape: try ints(size), stride: try ints(stride)))
        case "_rebuild_parameter":
            guard let t = args.first, case .tensor = t else {
                throw FormatError.invalid("_rebuild_parameter of something not a tensor")
            }
            return t
        default:
            throw FormatError.refused("calling \(module).\(name)")
        }
    }

    func ints(_ vs: [Value]) throws -> [int] {
        var out: [int] = []
        for v in vs {
            guard case .int(let i) = v else { throw FormatError.invalid("a size that is not an integer") }
            out.append(i)
        }
        return out
    }
}
