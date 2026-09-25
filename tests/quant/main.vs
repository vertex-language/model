// A quantized model's tensors decoded on every device by gpu/dtype, and
// compared bit for bit with what llama.cpp's own dequantizer makes of them
// (tests/oracle/gguf_dump.cpp -dequant): decoding is exact, so equal.
package main

import "fs"
import "gpu"
import "gpu/dtype"
import "model/gguf"

var failures = 0

func check(_ ok: bool, _ what: string) {
    print(ok ? "ok    \(what)" : "FAIL  \(what)")
    if !ok { failures += 1 }
}

func hex(_ v: uint64) -> string {
    let s = String(v, radix: 16)
    return String(repeating: "0", count: max(0, 16 - s.count)) + s
}

func hash(_ xs: [float32]) -> uint64 {
    var h: uint64 = 1469598103934665603
    for x in xs {
        var b = x.bitPattern
        for _ in 0..<4 {
            h ^= uint64(b & 0xFF)
            h = h &* 1099511628211
            b >>= 8
        }
    }
    return h
}

func decode(_ d: gpu.Device, _ f: gguf.File, _ t: gguf.TensorInfo) async throws -> [float32] {
    let w = try await d.Upload(f.Copy(t))
    let y = try await d.CreateBuffer(of: float32.self, count: t.Count)
    switch t.Type {
    case .Q4_0: try await dtype.Dequantize(w, dtype.Q4_0(), count: t.Count, into: y)
    case .Q8_0: try await dtype.Dequantize(w, dtype.Q8_0(), count: t.Count, into: y)
    default: throw gguf.FormatError.malformed("no decoder for \(t.Type.Name)")
    }
    return try await y.Download()
}

let f = try gguf.Open(fs.Path("testdata/stories15M-q4_0.gguf"))
let golden = try fs.ReadText(fs.Path("tests/quant/golden/stories15M-q4_0.dequant.txt"))
for line in golden.split(separator: "\n") {
    let parts = line.split(separator: " ")
    let name = String(parts[1])
    let want = String(parts[3])
    let t = f.Tensor(name)!
    for d in [gpu.CPU(), gpu.Default()] {
        let got = "fnv:" + hex(hash(try await decode(d, f, t)))
        check(got == want, "\(d.Name) \(name) \(t.Type.Name) x\(t.Count)")
    }
}
print(failures == 0 ? "all passed" : "\(failures) failed")
