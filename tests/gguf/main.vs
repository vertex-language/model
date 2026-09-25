// model/gguf against llama.cpp: each test model read by gguf.Open and
// dumped in the form tests/oracle/gguf_dump.cpp prints what llama.cpp's
// reader makes of it, compared line for line with that golden; then
// malformed files, each refused for the reason llama.cpp refuses it.
package main

import "fs"
import "model/gguf"

var failures = 0

func check(_ ok: bool, _ what: string) {
    print(ok ? "ok    \(what)" : "FAIL  \(what)")
    if !ok { failures += 1 }
}

// ---- the dump, in the oracle's form ----

func hex(_ v: uint64, _ digits: int) -> string {
    let s = String(v, radix: 16)
    return String(repeating: "0", count: max(0, digits - s.count)) + s
}

func fnv(_ h: inout uint64, _ b: uint8) {
    h ^= uint64(b)
    h = h &* 1099511628211
}

func fnvBytes(_ h: inout uint64, _ v: uint64, _ n: int) {
    for i in 0..<n { fnv(&h, uint8(truncatingIfNeeded: v >> uint64(8 * i))) }
}

func typeName(_ t: gguf.ValueType) -> string {
    switch t {
    case .U8: return "u8"
    case .I8: return "i8"
    case .U16: return "u16"
    case .I16: return "i16"
    case .U32: return "u32"
    case .I32: return "i32"
    case .F32: return "f32"
    case .Bool: return "bool"
    case .Text: return "str"
    case .Array: return "arr"
    case .U64: return "u64"
    case .I64: return "i64"
    case .F64: return "f64"
    }
}

func scalar(_ v: gguf.Value) -> string {
    switch v {
    case .U8(let x): return "\(x)"
    case .I8(let x): return "\(x)"
    case .U16(let x): return "\(x)"
    case .I16(let x): return "\(x)"
    case .U32(let x): return "\(x)"
    case .I32(let x): return "\(x)"
    case .U64(let x): return "\(x)"
    case .I64(let x): return "\(x)"
    case .Bool(let x): return x ? "true" : "false"
    case .F32(let x): return "f32:" + hex(uint64(x.bitPattern), 8)
    case .F64(let x): return "f64:" + hex(x.bitPattern, 16)
    case .Text(let s): return "\"" + s + "\""
    case .Array: return "?"
    }
}

func hashElement(_ h: inout uint64, _ v: gguf.Value) {
    switch v {
    case .U8(let x): fnv(&h, x)
    case .I8(let x): fnv(&h, uint8(bitPattern: x))
    case .Bool(let x): fnv(&h, x ? 1 : 0)
    case .U16(let x): fnvBytes(&h, uint64(x), 2)
    case .I16(let x): fnvBytes(&h, uint64(uint16(bitPattern: x)), 2)
    case .U32(let x): fnvBytes(&h, uint64(x), 4)
    case .I32(let x): fnvBytes(&h, uint64(uint32(bitPattern: x)), 4)
    case .F32(let x): fnvBytes(&h, uint64(x.bitPattern), 4)
    case .U64(let x): fnvBytes(&h, x, 8)
    case .I64(let x): fnvBytes(&h, uint64(bitPattern: x), 8)
    case .F64(let x): fnvBytes(&h, x.bitPattern, 8)
    case .Text(let s):
        for b in s.utf8 { fnv(&h, b) }
        fnv(&h, 0)
    case .Array: break
    }
}

func dump(_ f: gguf.File) -> [string] {
    var out = ["version \(f.Version)", "alignment \(f.Alignment)", "data \(f.DataOffset)"]
    for key in f.Keys {
        let v = f.Value(key)!
        var line = "kv \(key) \(typeName(v.Type)) "
        if case .Array(let et, let xs) = v {
            var h: uint64 = 1469598103934665603
            for x in xs { hashElement(&h, x) }
            line += "\(typeName(et)) x\(xs.count) fnv:" + hex(h, 16)
        } else {
            line += scalar(v)
        }
        out.append(line)
    }
    for t in f.Tensors {
        var h: uint64 = 1469598103934665603
        let p = f.Bytes(t)
        for i in 0..<t.Size { fnv(&h, p[i]) }
        out.append("tensor \(t.Name) \(t.Type.Name) at \(t.Offset) size \(t.Size) fnv:" + hex(h, 16))
    }
    return out
}

let here = fs.Path("tests/gguf/golden")
for name in ["stories260K", "stories15M-q4_0"] {
    let f = try gguf.Open(fs.Path("testdata/\(name).gguf"))
    let got = dump(f)
    let want = try fs.ReadText(here.Join(fs.Path("\(name).txt"))).split(separator: "\n").map { String($0) }
    var same = got.count == want.count
    for i in 0..<min(got.count, want.count) {
        if got[i] != want[i] {
            if same { print("      first difference, line \(i + 1):\n      got  \(got[i])\n      want \(want[i])") }
            same = false
        }
    }
    check(same, "\(name): \(got.count) lines, as llama.cpp reads it")
}

// ---- the typed accessors, on the quantized model ----

let q = try gguf.Open(fs.Path("testdata/stories15M-q4_0.gguf"))
check(q.Architecture == "llama", "Architecture")
check(q.Integer("llama.block_count") == 6 && q.Integer("llama.embedding_length") == 288, "Integer")
check(q.Number("llama.attention.layer_norm_rms_epsilon") == float64(float32(1e-5)), "Number of an f32")
check(q.Texts("tokenizer.ggml.tokens")?.count == 32000 && q.Texts("tokenizer.ggml.tokens")?[1] == "<s>", "Texts")
check(q.Numbers("tokenizer.ggml.scores")?.count == 32000, "Numbers")
check(q.Integers("tokenizer.ggml.token_type")?[0] == 2, "Integers")
check(q.Text("no.such.key") == nil && q.Integer("general.architecture") == nil, "a missing or mistyped key is nil")
let embd = q.Tensor("token_embd.weight")!
check(embd.Type == .Q4_0 && embd.Shape == [288, 32000] && embd.Count == 288 * 32000, "token_embd is q4_0, 288 x 32000")
check(embd.Size == embd.Type.RowSize(288) * 32000, "its size is 32000 rows of 9 q4_0 blocks")
check(q.Tensor("output.weight")!.Type == .Q8_0, "output is q8_0")

// ---- malformed files ----

func le32(_ v: uint32) -> [uint8] {
    return [uint8(truncatingIfNeeded: v), uint8(truncatingIfNeeded: v >> 8), uint8(truncatingIfNeeded: v >> 16), uint8(truncatingIfNeeded: v >> 24)]
}

func le64(_ v: uint64) -> [uint8] {
    return le32(uint32(truncatingIfNeeded: v)) + le32(uint32(truncatingIfNeeded: v >> 32))
}

func text(_ s: string) -> [uint8] {
    return le64(uint64(s.utf8.count)) + Array(s.utf8)
}

func header(_ tensors: uint64, _ kvs: uint64) -> [uint8] {
    return Array("GGUF".utf8) + le32(3) + le64(tensors) + le64(kvs)
}

let dir = fs.Path("/tmp/vertex_model_gguf_test")
try? fs.RemoveAll(dir)
try fs.CreateDir(dir)
var n = 0

func refused(_ bytes: [uint8], _ what: string, _ reason: string) {
    n += 1
    let path = dir.Join(fs.Path("bad\(n).gguf"))
    do {
        try fs.WriteFile(path, bytes)
        _ = try gguf.Open(path)
        check(false, "refused: \(what)")
    } catch let e as gguf.FormatError {
        let ok = e.Message.contains(reason)
        if !ok { print("      said: \(e.Message)") }
        check(ok, "refused: \(what)")
    } catch {
        check(false, "refused: \(what) (\(error))")
    }
}

let u32kv = text("a") + le32(4) + le32(7)
refused(Array("GGUX".utf8) + le32(3) + le64(0) + le64(0), "a bad magic", "not a GGUF file")
refused(Array("GGUF".utf8) + le32(1) + le64(0) + le64(0), "version 1", "v1")
refused(Array("GGUF".utf8) + le32(4) + le64(0) + le64(0), "version 4", "newer")
refused(Array("GGUF".utf8) + le32(0x03000000) + le64(0) + le64(0), "the other endianness", "endianness")
refused(header(0, 1) + text("a") + le32(4), "a truncated value", "past the end")
refused(header(0, 1) + text("") + le32(4) + le32(7), "an empty key", "is empty")
refused(header(0, 2) + u32kv + u32kv, "a duplicate key", "twice")
refused(header(0, 1) + text("a") + le32(13) + le32(7), "an unknown value type", "unknown type")
refused(header(0, 1) + text("a") + le32(9) + le32(9) + le64(0), "a nested array", "not allowed")
refused(header(0, 1) + text("a") + le32(9) + le32(4) + le64(1 << 40), "an array longer than the file", "more than the file holds")
refused(header(0, 1) + text("general.alignment") + le32(4) + le32(24), "an alignment not a power of 2", "power of 2")
refused(header(0, 1) + text("general.alignment") + le32(5) + le32(32), "an alignment not a u32", "must be a u32")
refused(header(1 << 40, 0), "more tensors than the file holds", "more than the file holds")

func tensor(_ name: string, _ dims: [uint64], _ type: uint32, _ offset: uint64) -> [uint8] {
    var b = text(name) + le32(uint32(dims.count))
    for d in dims { b += le64(d) }
    return b + le32(type) + le64(offset)
}

func withData(_ b: [uint8], _ n: int) -> [uint8] {
    var out = b
    while out.count % 32 != 0 { out.append(0) }
    return out + [uint8](repeating: 0, count: n)
}

refused(withData(header(1, 0) + tensor("t", [1, 1, 1, 1, 1], 0, 0), 4), "five dimensions", "dimensions")
refused(withData(header(1, 0) + tensor("t", [4], 99, 0), 16), "an unknown tensor type", "unknown type 99")
refused(withData(header(1, 0) + tensor("t", [33], 2, 0), 64), "a q4_0 row not a multiple of 32", "not a multiple")
refused(withData(header(2, 0) + tensor("t", [4], 0, 0) + tensor("t", [4], 0, 32), 64), "a duplicate tensor", "twice")
refused(withData(header(2, 0) + tensor("a", [4], 0, 0) + tensor("b", [4], 0, 16), 64), "an offset not padded", "was expected")
refused(withData(header(1, 0) + tensor("t", [64], 0, 0), 16), "a tensor past the end (llama.cpp finds it on loading)", "past the end")
refused(withData(header(1, 0) + tensor(String(repeating: "x", count: 64), [4], 0, 0), 16), "a 64-byte tensor name", "over 63")

// And a small file that is well-formed: two tensors, padded, read back.
var good = header(2, 2) + text("general.architecture") + le32(8) + text("toy")
good += text("toy.flag") + le32(7) + [2]
good += tensor("a", [3], 0, 0) + tensor("b", [2, 2], 30, 32)
good = withData(good, 0)
for v in [float32(1.5), -2, 0.25] { good += le32(v.bitPattern) }
good += [uint8](repeating: 0, count: 20)
good += [0x80, 0x3F, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40]
let path = dir.Join(fs.Path("good.gguf"))
try fs.WriteFile(path, good)
let g = try gguf.Open(path)
check(g.Architecture == "toy" && g.Flag("toy.flag") == true, "a toy file's metadata, a bool of 2 read as true as llama.cpp does")
let a = g.Tensor("a")!, b = g.Tensor("b")!
check(a.Type == .F32 && a.Size == 12 && float32(bitPattern: uint32(g.Bytes(a)[0]) | uint32(g.Bytes(a)[1]) << 8 | uint32(g.Bytes(a)[2]) << 16 | uint32(g.Bytes(a)[3]) << 24) == 1.5, "tensor a's first f32")
check(b.Type == .BF16 && b.Shape == [2, 2] && g.Copy(b) == [0x80, 0x3F, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40], "tensor b's bf16 bytes, 32 bytes in")

try? fs.RemoveAll(dir)
print(failures == 0 ? "all passed" : "\(failures) failed")
