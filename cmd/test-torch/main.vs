// model/torch against torch.load: every tensor's name, dtype, shape and the
// FNV-1a hash of its bytes, line for line as testdata/oracle/torch_dump.py
// prints them (golden/), for the fixtures in testdata/torch and, when they
// are in the Hugging Face cache, Kokoro-82M's weights and a voice. And two
// refusals: a pickle that calls os.system, and a view that is not
// contiguous. Kokoro's directory also opens as a model.Checkpoint, whose
// tensors are the same bytes.
package main

import (
    "fs"
    "model"
    "model/torch"
    "os/env"
)

var failures = 0

func check(_ ok: bool, _ what: string) {
    print(ok ? "ok    \(what)" : "FAIL  \(what)")
    if !ok { failures += 1 }
}

func fnv(_ p: UnsafePointer<uint8>, _ n: int) -> uint64 {
    var h: uint64 = 0xcbf29ce484222325
    var i = 0
    while i < n {
        h = (h ^ uint64(p[i])) &* 0x100000001b3
        i += 1
    }
    return h
}

func hex16(_ v: uint64) -> string {
    let s = string(v, radix: 16)
    return string(repeating: "0", count: 16 - s.count) + s
}

// dump is what torch_dump.py prints for a file.
func dump(_ f: torch.File) -> [string] {
    var out: [string] = []
    for t in f.Tensors {
        var shape = t.Shape.map { "\($0)" }.joined(separator: "x")
        if shape.isEmpty { shape = "scalar" }
        let name = t.Name.isEmpty ? "<root>" : t.Name
        out.append("\(name) \(t.DType.Name) \(shape) \(hex16(fnv(f.Bytes(t), t.Size)))")
    }
    return out
}

func against(_ path: fs.Path, _ golden: string) {
    do {
        let f = try torch.Open(path)
        let want = try fs.ReadText(fs.Path("cmd/test-torch/golden/" + golden)).split(separator: "\n").map { String($0) }
        let got = dump(f)
        var same = got.count == want.count
        var i = 0
        while same && i < got.count {
            if got[i] != want[i] {
                same = false
                print("      got  \(got[i])\n      want \(want[i])")
            }
            i += 1
        }
        check(same, "\(golden): \(got.count) tensors as torch.load reads them")
    } catch {
        check(false, "\(path.String()): \(error)")
    }
}

// throughCheckpoint opens a directory with model.Open, which finds its
// PyTorch weights (and not the voices/ beside them), and dumps each
// tensor from what Copy hands back.
func throughCheckpoint(_ dir: fs.Path, _ golden: string) {
    do {
        let ck = try model.Open(dir)
        let want = try fs.ReadText(fs.Path("cmd/test-torch/golden/" + golden)).split(separator: "\n").map { String($0) }
        var got: [string] = []
        for name in ck.Names {
            guard let t = ck.Info(name) else { continue }
            var shape = t.Shape.map { "\($0)" }.joined(separator: "x")
            if shape.isEmpty { shape = "scalar" }
            let bytes = ck.Copy(t)
            let h = bytes.withUnsafeBufferPointer { fnv($0.baseAddress!, $0.count) }
            got.append("\(name) \(t.SourceType) \(shape) \(hex16(h))")
        }
        check(ck.Format == .torch && ck.Architecture == "" && got == want,
              "model.Open(Kokoro-82M): \(ck.Format.Name), \(got.count) tensors, the same bytes")
    } catch {
        check(false, "model.Open(\(dir.String())): \(error)")
    }
}

against(fs.Path("testdata/torch/small.pt"), "small.txt")
against(fs.Path("testdata/torch/lone.pt"), "lone.txt")

do {
    _ = try torch.Open(fs.Path("testdata/torch/evil.pt"))
    check(false, "a pickle that calls os.system is refused")
} catch {
    check("\(error)".contains("refused: the global") && "\(error)".contains("system"),
          "a pickle that calls os.system is refused: \(error)")
}
do {
    _ = try torch.Open(fs.Path("testdata/torch/transposed.pt"))
    check(false, "a transposed view is refused")
} catch {
    check("\(error)".contains("not contiguous"), "a transposed view is refused: \(error)")
}

// Kokoro-82M, where hub (or huggingface_hub) put it.
let home = env.Get("HOME") ?? ""
let snapshots = fs.Path(home + "/.cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots")
if let snaps = try? fs.ReadDir(snapshots), let snap = snaps.first {
    let dir = snapshots / snap.Name
    against(dir / "kokoro-v1_0.pth", "kokoro-v1_0.txt")
    against(dir / "voices/af_heart.pt", "af_heart.txt")
    throughCheckpoint(dir, "kokoro-v1_0.txt")
} else {
    print("skip  Kokoro-82M is not in the Hugging Face cache (vsc run hub -- download hf.co/hexgrad/Kokoro-82M)")
}

print(failures == 0 ? "all passed" : "\(failures) failed")
