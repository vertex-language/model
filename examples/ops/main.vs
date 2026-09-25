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

// 7B-class shapes, reported as the bandwidth the weights were read at.
func bandwidth(_ name: string, _ bytes: int, _ body: () async throws -> Void) async throws {
    try await body()
    _ = try await d.CreateBuffer(of: float32.self, count: 1).Download()
    let t = time.Instant.Now()
    for _ in 0..<20 { try await body() }
    _ = try await d.CreateBuffer(of: float32.self, count: 1).Download()
    let us = t.Elapsed().AsMicroseconds() / 20
    print("\(name): \(us)us, \(int64(bytes) / max(us, 1) / 1000) GB/s")
}
let big = 14336 * 4096
let w4 = try await d.Upload([uint8](repeating: 0x5A, count: big / 32 * 18))
let w8 = try await d.Upload([uint8](repeating: 0x21, count: big / 32 * 34))
let xk = try await d.Upload([float32](repeating: 0.01, count: 14336))
let yk = try d.CreateBuffer(of: float32.self, count: 14336)
try await bandwidth("gemv q4_0 4096x14336", big / 32 * 18, { try await linalg.Gemv(w4, dtype.Q4_0(), xk, into: yk.Slice(from: 0, count: 4096), m: 4096, k: 14336) })
try await bandwidth("gemv q4_0 14336x4096", big / 32 * 18, { try await linalg.Gemv(w4, dtype.Q4_0(), xk.Slice(from: 0, count: 4096), into: yk, m: 14336, k: 4096) })
try await bandwidth("gemv q8_0 4096x14336", big / 32 * 34, { try await linalg.Gemv(w8, dtype.Q8_0(), xk, into: yk.Slice(from: 0, count: 4096), m: 4096, k: 14336) })
let wk4 = try await d.Upload([uint8](repeating: 0x11, count: big / 256 * 144))
let wk6 = try await d.Upload([uint8](repeating: 0x11, count: big / 256 * 210))
try await bandwidth("gemv q4_K 4096x14336", big / 256 * 144, { try await linalg.Gemv(wk4, dtype.Q4_K(), xk, into: yk.Slice(from: 0, count: 4096), m: 4096, k: 14336) })
try await bandwidth("gemv q4_K 14336x4096", big / 256 * 144, { try await linalg.Gemv(wk4, dtype.Q4_K(), xk.Slice(from: 0, count: 4096), into: yk, m: 14336, k: 4096) })
try await bandwidth("gemv q6_K 4096x14336", big / 256 * 210, { try await linalg.Gemv(wk6, dtype.Q6_K(), xk, into: yk.Slice(from: 0, count: 4096), m: 4096, k: 14336) })

// stories110M's shapes (dim 768, 12 heads of 64, hidden 2048, q4_K and q6_K).
print("-- 110M")
let x768 = xk.Slice(from: 0, count: 768), x2048 = xk.Slice(from: 0, count: 2048)
let w768 = try await d.Upload([float32](repeating: 1, count: 768))
let o768 = try d.CreateBuffer(of: float32.self, count: 768)
let kc12 = try await d.Upload([float32](repeating: 0.1, count: 12 * 1024 * 64))
try await timed("gemv q4_K 1536x768 (q,k)", { try await linalg.Gemv(wk4, dtype.Q4_K(), x768, into: yk.Slice(from: 0, count: 1536), m: 1536, k: 768) })
try await timed("gemv q6_K 768x768 (v)", { try await linalg.Gemv(wk6, dtype.Q6_K(), x768, into: o768, m: 768, k: 768) })
try await timed("gemv q4_K 768x768 (o)", { try await linalg.Gemv(wk4, dtype.Q4_K(), x768, into: o768, m: 768, k: 768) })
try await timed("gemv q4_K 4096x768 (gate, up)", { try await linalg.Gemv(wk4, dtype.Q4_K(), x768, into: yk.Slice(from: 0, count: 4096), m: 4096, k: 768) })
try await timed("gemv q6_K 768x2048 (down)", { try await linalg.Gemv(wk6, dtype.Q6_K(), x2048, into: o768, m: 768, k: 2048) })
try await timed("gemv q6_K 32000x768 (output)", { try await linalg.Gemv(wk6, dtype.Q6_K(), x768, into: yk.Slice(from: 0, count: 14336), m: 14336, k: 768) })
try await timed("rmsnorm 768", { try await neural.RMSNorm(o768, weight: w768, into: o768, rows: 1, cols: 768, eps: 1e-5) })
try await timed("rope 24x64", { try await neural.RoPE(yk.Slice(from: 0, count: 1536), position: 7, heads: 24, dim: 64) })
try await timed("attention 12x64, 50 keys", { try await attention.Forward(q: o768, k: kc12, v: kc12, into: o768, attention.Shape(heads: 12, queries: 1, keys: 50, headDim: 64, keyCapacity: 1024), mask: .Causal) })
try await timed("gated 2048", { try await neural.Gated(x2048, gate: x2048, .SiLU, into: yk.Slice(from: 0, count: 2048)) })
try await timed("dequantize q4_K 768", { try await dtype.Dequantize(wk4, dtype.Q4_K(), count: 768, into: o768) })
