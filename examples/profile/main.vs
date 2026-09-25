// Where a decoded token's time goes: encoding its launches, the device
// running them, and reading the logits back.
package main

import "fs"
import "gpu"
import "model/llama"
import "time"

for d in [gpu.Default()] {
    let m = try await llama.Model.Load(fs.Path("testdata/stories15M-q4_0.gguf"), on: d)
    _ = try await m.Forward(1, position: 0).Download()
    let n = 50
    var encode: int64 = 0, run: int64 = 0, read: int64 = 0, pick: int64 = 0
    for p in 1...n {
        var t = time.Instant.Now()
        let logits = try await m.Forward(9038, position: p)
        encode += t.Elapsed().AsMicroseconds()
        t = time.Instant.Now()
        _ = logits._elements
        run += t.Elapsed().AsMicroseconds()
        t = time.Instant.Now()
        let host = try await logits.Download()
        read += t.Elapsed().AsMicroseconds()
        t = time.Instant.Now()
        _ = llama.Argmax(host)
        pick += t.Elapsed().AsMicroseconds()
    }
    print("\(d.Name) per token: encode \(encode / int64(n))us, run \(run / int64(n))us, download \(read / int64(n))us, argmax \(pick / int64(n))us")
}
