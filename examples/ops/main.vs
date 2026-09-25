// Each operation of a decoded token alone, at stories15M's sizes: its time
// on the device, 200 in a row and one wait.
package main

import "gpu"
import "gpu/attention"
import "gpu/dtype"
import "gpu/linalg"
import "gpu/neural"
import "tensor"
import "time"

let d = gpu.Default()
func timed(_ name: string, _ body: () async throws -> Void) async throws {
    try await body()
    _ = try await d.CreateBuffer(of: float32.self, count: 1).Download()
    let t = time.Instant.Now()
    for _ in 0..<200 { try await body() }
    let probe = try d.CreateBuffer(of: float32.self, count: 1)
    _ = try await probe.Download()
    print("\(name): \(t.Elapsed().AsMicroseconds() / 200)us")
}
let x = try await d.Upload([float32](repeating: 0.01, count: 768))
let y = try d.CreateBuffer(of: float32.self, count: 32000)
let x288 = x.Slice(from: 0, count: 288)
let q4 = try await d.Upload([uint8](repeating: 0x11, count: 768 * 288 / 32 * 18))
let q8 = try await d.Upload([uint8](repeating: 0x11, count: 32000 * 288 / 32 * 34))
let w = try await d.Upload([float32](repeating: 1, count: 288))
let q = try await d.Upload([float32](repeating: 0.1, count: 288))
let kc = try await d.Upload([float32](repeating: 0.1, count: 6 * 128 * 48))
let o = try d.CreateBuffer(of: float32.self, count: 288)
try await timed("gemv q4_0 288x288", { try await linalg.Gemv(q4, dtype.Q4_0(), x288, into: y.Slice(from: 0, count: 288), m: 288, k: 288) })
try await timed("gemv q4_0 768x288", { try await linalg.Gemv(q4, dtype.Q4_0(), x288, into: y.Slice(from: 0, count: 768), m: 768, k: 288) })
try await timed("gemv q4_0 288x768", { try await linalg.Gemv(q4, dtype.Q4_0(), x, into: y.Slice(from: 0, count: 288), m: 288, k: 768) })
try await timed("gemv q8_0 32000x288", { try await linalg.Gemv(q8, dtype.Q8_0(), x288, into: y, m: 32000, k: 288) })
try await timed("rmsnorm 288", { try await neural.RMSNorm(q, weight: w, into: o, rows: 1, cols: 288, eps: 1e-5) })
try await timed("rope 6x48", { try await neural.RoPE(o, position: 7, heads: 6, dim: 48) })
try await timed("attention 6 heads, 50 keys", { try await attention.Forward(q: q, k: kc, v: kc, into: o, attention.Shape(heads: 6, queries: 1, keys: 50, headDim: 48, keyCapacity: 128), mask: .Causal) })
try await timed("add 288", { try await tensor.Add(q, w, into: o) })
try await timed("gated 768", { try await neural.Gated(x, gate: x, .SiLU, into: y.Slice(from: 0, count: 768)) })
try await timed("dequantize 288", { try await dtype.Dequantize(q4, dtype.Q4_0(), count: 288, into: o) })
