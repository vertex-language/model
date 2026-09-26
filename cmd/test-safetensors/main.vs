package main

import (
    "fs"
    "model/safetensors"
)

// testdata/tiny-llama (hf.co/hf-internal-testing/tiny-random-LlamaForCausalLM,
// written by Python's safetensors) dumped as testdata/oracle/safetensors_dump.py
// dumps it; a round trip through Encode; and malformed files, each refused.

var failures = 0

func check(_ ok: bool, _ msg: string) {
    if ok {
        print("ok    \(msg)")
    } else {
        print("FAIL  \(msg)")
        failures += 1
    }
}

func fnv1a(_ p: UnsafePointer<uint8>, _ n: int) -> uint64 {
    var h: uint64 = 0xcbf29ce484222325
    for i in 0..<n {
        h ^= uint64(p[i])
        h = h &* 0x100000001b3
    }
    return h
}

func hex16(_ v: uint64) -> string {
    let digits: [Character] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"]
    var s = ""
    var i = 60
    while i >= 0 {
        s.append(digits[int((v >> uint64(i)) & 0xF)])
        i -= 4
    }
    return s
}

func dump(_ f: safetensors.File) -> [string] {
    var out: [string] = []
    for t in f.Tensors {
        let shape = t.Shape.map { "\($0)" }.joined(separator: ",")
        let b = t.Offset - f.DataOffset
        out.append("\(t.Name) \(t.DType.Name) [\(shape)] \(b) \(b + t.Size) \(hex16(fnv1a(f.Bytes(t), t.Size)))")
    }
    for k in f.Metadata.keys.sorted() {
        out.append("meta \(k)=\(f.Metadata[k]!)")
    }
    return out
}

// refused writes bytes to a file, opens it, and reports whether it was
// refused with a message holding why.
func refused(_ dir: fs.Path, _ name: string, _ bytes: [uint8], _ why: string) {
    let p = dir / "\(name).safetensors"
    do {
        try fs.WriteFile(p, bytes)
        _ = try safetensors.Open(p)
        check(false, "\(name): refused")
    } catch {
        let msg = "\(error)"
        check(msg.contains(why), "\(name): refused (\(msg))")
    }
}

// raw is a file of a header given as text and data bytes.
func raw(_ header: string, _ data: [uint8]) -> [uint8] {
    let h = [uint8](header.utf8)
    var out: [uint8] = []
    var n = uint64(h.count)
    for _ in 0..<8 {
        out.append(uint8(truncatingIfNeeded: n))
        n >>= 8
    }
    return out + h + data
}

func main() -> int32 {
    // The golden dump of a file Python wrote.
    do {
        let f = try safetensors.Open(fs.Path("testdata/tiny-llama/model.safetensors"))
        let want = try fs.ReadText(fs.Path("cmd/test-safetensors/golden/tiny-llama.txt")).split(separator: "\n").map { String($0) }
        let got = dump(f)
        check(got == want, "tiny-llama: \(got.count) lines as the oracle dumps them")
        if got != want {
            for i in 0..<min(got.count, want.count) where got[i] != want[i] {
                print("  got  \(got[i])\n  want \(want[i])")
                break
            }
        }
        check(f.Tensor("model.layers.1.self_attn.q_proj.weight")?.Shape == [16, 16], "lookup by name")
        check(f.Tensor("no.such.tensor") == nil, "a missing name is nil")
    } catch {
        check(false, "tiny-llama: \(error)")
    }

    do {
        let dir = try fs.TempDir(prefix: "safetensors_")
        // Round trip, with a zero-size tensor and metadata.
        let entries = [
            safetensors.Entry(name: "b", dtype: .BF16, shape: [2, 3], bytes: (0..<12).map { uint8($0) }),
            safetensors.Entry(name: "a", dtype: .F32, shape: [1], bytes: [0, 0, 128, 63]),
            safetensors.Entry(name: "empty", dtype: .I64, shape: [0, 4], bytes: []),
            safetensors.Entry(name: "flags", dtype: .BOOL, shape: [3], bytes: [1, 0, 1]),
        ]
        let p = dir / "rt.safetensors"
        try safetensors.Save(entries, to: p, metadata: ["format": "pt", "who": "vertex"])
        let f = try safetensors.Open(p)
        check(f.DataOffset % 8 == 0, "the header is padded to 8 bytes")
        check(f.Tensors.map { $0.Name } == ["b", "a", "empty", "flags"], "round trip: names in byte order")
        check(f.Copy(f.Tensor("b")!) == entries[0].Bytes && f.Tensor("b")!.DType == .BF16 && f.Tensor("b")!.Shape == [2, 3], "round trip: bf16 bytes, dtype, shape")
        check(f.Copy(f.Tensor("a")!) == [0, 0, 128, 63], "round trip: f32")
        check(f.Tensor("empty")!.Size == 0 && f.Tensor("empty")!.Count == 0, "round trip: a zero-size tensor")
        check(f.Metadata == ["format": "pt", "who": "vertex"], "round trip: metadata")
        do {
            _ = try safetensors.Encode([safetensors.Entry(name: "x", dtype: .F32, shape: [2], bytes: [1, 2, 3])])
            check(false, "Encode refuses bytes that do not fit the shape")
        } catch {
            check(true, "Encode refuses bytes that do not fit the shape")
        }

        // Malformed files, as the Rust crate refuses them.
        let four: [uint8] = [0, 0, 128, 63]
        refused(dir, "short", [1, 2, 3], "too short")
        refused(dir, "huge-header", [0, 0, 0, 0, 1, 0, 0, 0], "past the limit")
        refused(dir, "header-past-end", [100, 0, 0, 0, 0, 0, 0, 0, 123, 125], "runs past the file")
        refused(dir, "not-an-object", raw("[1, 2]", []), "start with '{'")
        refused(dir, "not-json", raw("{\"a\": ", []), "not JSON")
        refused(dir, "unknown-dtype", raw("{\"a\":{\"dtype\":\"F33\",\"shape\":[1],\"data_offsets\":[0,4]}}", four), "dtype")
        refused(dir, "wrong-size", raw("{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}}", four), "which is 8")
        refused(dir, "past-data", raw("{\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}", four), "run past the data")
        refused(dir, "gap", raw("{\"a\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[2,4]}}", four), "a gap or an overlap")
        refused(dir, "overlap", raw("{\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]},\"b\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[2,4]}}", four), "a gap or an overlap")
        refused(dir, "trailing", raw("{\"a\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]}}", four), "runs to 4")
        refused(dir, "negative-shape", raw("{\"a\":{\"dtype\":\"U8\",\"shape\":[-1],\"data_offsets\":[0,4]}}", four), "non-negative")
        refused(dir, "bad-offsets", raw("{\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[4,0]}}", four), "data_offsets")
        refused(dir, "metadata-not-strings", raw("{\"__metadata__\":{\"n\":1}}", []), "not a string")
        refused(dir, "half-byte", raw("{\"a\":{\"dtype\":\"F4\",\"shape\":[3],\"data_offsets\":[0,2]}}", [0, 0]), "whole bytes")
        try fs.RemoveAll(dir)
    } catch {
        check(false, "temp files: \(error)")
    }

    if failures == 0 {
        print("\n=== all model/safetensors checks passed ===")
        return 0
    }
    print("\n\(failures) failed")
    return 1
}
