// model/llama against llama.cpp: "Once upon a time" through each test
// model a token at a time, on every device -- the logits of each step
// where llama.cpp's top five are, and its greedy continuation, token for
// token (golden/, from tests/oracle/llama_run.cpp).
//
// llama.cpp is not exact: at the f32 model's first step its logits are
// 2.5e-4 from the float64 answer, where these are 1e-6 from it (float32
// rounding). So that first step is also checked against float64 math
// written out here, tightly, and llama.cpp's logits loosely: 0.01 for the
// f32 model, and for the q4_0 model 0.25, since llama.cpp's CPU path also
// rounds each activation to q8_0 before its dot product with q4_0 weights.
package main

import "fs"
import "gpu"
import "model/gguf"
import "model/llama"

@_silgen_name("exp") func cExp(_ x: float64) -> float64

// exactFirstStep is the f32 model's logits for <s> at position 0 in
// float64: one key, so attention is V; RoPE at 0 turns nothing.
func exactFirstStep(_ f: gguf.File, _ c: llama.Config) -> [float64] {
    func w(_ name: string) -> [float64] {
        let t = f.Tensor(name)!
        let p = f.Bytes(t)
        var out: [float64] = []
        for i in 0..<t.Count {
            let b = uint32(p[4 * i]) | uint32(p[4 * i + 1]) << 8 | uint32(p[4 * i + 2]) << 16 | uint32(p[4 * i + 3]) << 24
            out.append(float64(float32(bitPattern: b)))
        }
        return out
    }
    func mv(_ m: [float64], _ x: [float64], _ rows: int) -> [float64] {
        var y: [float64] = []
        for r in 0..<rows {
            var s: float64 = 0
            for i in 0..<x.count { s += m[r * x.count + i] * x[i] }
            y.append(s)
        }
        return y
    }
    func rms(_ x: [float64], _ g: [float64]) -> [float64] {
        var ms: float64 = 0
        for v in x { ms += v * v }
        let r = 1 / (ms / float64(x.count) + float64(c.Eps)).squareRoot()
        return (0..<x.count).map { x[$0] * r * g[$0] }
    }
    let dim = c.Dim, hd = c.HeadDim
    var x = Array(w("token_embd.weight")[dim..<(2 * dim)])
    for l in 0..<c.Layers {
        let p = "blk.\(l)."
        let v = mv(w(p + "attn_v.weight"), rms(x, w(p + "attn_norm.weight")), c.KVHeads * hd)
        var o: [float64] = []
        for h in 0..<c.Heads { for i in 0..<hd { o.append(v[(h / (c.Heads / c.KVHeads)) * hd + i]) } }
        let a = mv(w(p + "attn_output.weight"), o, dim)
        for i in 0..<dim { x[i] += a[i] }
        let h2 = rms(x, w(p + "ffn_norm.weight"))
        let g = mv(w(p + "ffn_gate.weight"), h2, c.Hidden), u = mv(w(p + "ffn_up.weight"), h2, c.Hidden)
        let dn = mv(w(p + "ffn_down.weight"), (0..<c.Hidden).map { g[$0] / (1 + cExp(-g[$0])) * u[$0] }, dim)
        for i in 0..<dim { x[i] += dn[i] }
    }
    return mv(w("output.weight"), rms(x, w("output_norm.weight")), c.Vocab)
}

var failures = 0

func check(_ ok: bool, _ what: string) {
    print(ok ? "ok    \(what)" : "FAIL  \(what)")
    if !ok { failures += 1 }
}

for (name, tolerance) in [("stories260K", 0.01), ("stories15M-q4_0", 0.25)] {
    let golden = try fs.ReadText(fs.Path("tests/llama/golden/\(name).txt")).split(separator: "\n")
    var steps: [(int, [(int, float64)])] = []
    var greedy: [int] = []
    for line in golden {
        let f = line.split(separator: " ")
        if f[0] == "greedy" {
            greedy = f.dropFirst().map { int(String($0))! }
            continue
        }
        var top: [(int, float64)] = []
        for i in 5..<10 {
            let kv = f[i].split(separator: ":")
            top.append((int(String(kv[0]))!, float64(String(kv[1]))!))
        }
        steps.append((int(String(f[3]))!, top))
    }
    for d in [gpu.CPU(), gpu.Default()] {
        let m = try await llama.Model.Load(fs.Path("testdata/\(name).gguf"), on: d)
        var worst: float64 = 0
        var argmaxSame = true
        var position = 0
        for (token, top) in steps.prefix(4) {
            let logits = try await m.Forward(token, position: position).Download()
            position += 1
            for (id, want) in top {
                worst = max(worst, (float64(logits[id]) - want).magnitude)
            }
            if llama.Argmax(logits) != top[0].0 { argmaxSame = false }
        }
        check(argmaxSame, "\(d.Name) \(name): the prompt's next tokens are llama.cpp's")
        if name == "stories260K" {
            let f = try gguf.Open(fs.Path("testdata/\(name).gguf"))
            let exact = exactFirstStep(f, m.Config)
            let m3 = try await llama.Model.Load(fs.Path("testdata/\(name).gguf"), on: d)
            let first = try await m3.Forward(1, position: 0).Download()
            var off: float64 = 0
            for i in 0..<exact.count { off = max(off, (float64(first[i]) - exact[i]).magnitude) }
            check(off < 1e-4, "\(d.Name) \(name): the first step's \(exact.count) logits within 1e-4 of float64 (at most \(off))")
        }
        check(worst <= tolerance, "\(d.Name) \(name): logits within \(tolerance) of llama.cpp's (at most \(worst))")
        let m2 = try await llama.Model.Load(fs.Path("testdata/\(name).gguf"), on: d)
        let got = try await m2.Generate("Once upon a time", tokens: greedy.count)
        check(got == greedy, "\(d.Name) \(name): \(greedy.count) greedy tokens, as llama.cpp's")
        if got != greedy { print("      got  \(got)\n      want \(greedy)") }
        print("      \"Once upon a time\(m2.Tokenizer.Decode(got))\"")
    }
}
print(failures == 0 ? "all passed" : "\(failures) failed")
