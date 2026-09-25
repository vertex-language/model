// Decode speed: a model's load time, and the tokens a second it generates
// greedily after a short prompt, on each device.
//
//   MODEL=testdata/stories110M-q4_k_m.gguf TOKENS=100 vsc run bench
package main

import "fs"
import "gpu"
import "model/llama"
import "os/env"
import "time"

let path = env.Get("MODEL") ?? "testdata/stories15M-q4_0.gguf"
let n = int(env.Get("TOKENS") ?? "100") ?? 100
for d in [gpu.Default(), gpu.CPU()] {
    var t = time.Instant.Now()
    let m = try await llama.Model.Load(fs.Path(path), on: d)
    let load = t.Elapsed()
    t = time.Instant.Now()
    let out = try await m.Generate("Once upon a time", tokens: n)
    let gen = t.Elapsed()
    let perSecond = float64(out.count + 5) / gen.AsSeconds()
    print("\(d.Name): load \(load), \(out.count) tokens in \(gen): \(int(perSecond)) tokens/s")
}
