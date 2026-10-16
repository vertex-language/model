// registry: the list of known models, by name.
//
//     registry list [task]              every model, or a task's
//     registry show qwen3:8b            its repository, commit, quants and files
//     registry pull qwen3:8b-q8_0       download it into the Hugging Face cache
//     registry probe [ref]... [--all]   the first bytes of each file, anonymously
package main

import (
    "model/registry"
    "remote/hub"
)

func usage() -> int32 {
    print("usage: registry list [task]")
    print("       registry show <ref>")
    print("       registry pull <ref>... [--quiet]")
    print("       registry probe <ref>... | --all [--every-quant]")
    print("")
    print("ref:   name, name:tag, name:tag-quant   (qwen3, qwen3:8b, qwen3:8b-q8_0)")
    return 2
}

func size(_ n: int64) -> string {
    let units = ["B", "KB", "MB", "GB", "TB"]
    var v = float64(n)
    var u = 0
    while v >= 1000 && u < units.count - 1 {
        v /= 1000
        u += 1
    }
    if u == 0 {
        return "\(n) B"
    }
    let tenths = int64(v * 10 + 0.5)
    return "\(tenths / 10).\(tenths % 10) \(units[u])"
}

func pad(_ s: string, _ n: int) -> string {
    var out = s
    while out.count < n {
        out += " "
    }
    return out
}

func list(_ task: string?) {
    for e in registry.Entries() where task == nil || e.Task == task! {
        let def = e.QuantNamed(e.Default)
        let pulled = def != nil && registry.IsPulled(registry.Pick(Entry: e, Quant: def!))
        print("\(pad(e.ID, 34)) \(pad(e.Task, 9)) \(pad(e.Arch, 14)) \(pad(size(def?.Size ?? 0), 9)) \(pulled ? "pulled" : "      ") \(e.Latest ? "latest" : "")")
    }
}

func show(_ ref: string) throws {
    let p = try registry.Find(ref)
    let e = p.Entry
    print("\(p.ID)")
    print("  task      \(e.Task)")
    print("  source    hf.co/\(e.Source)")
    print("  from      hf.co/\(e.Repo)@\(e.Revision)")
    print("  arch      \(e.Arch)")
    print("  license   \(e.License)")
    print("  quants")
    for q in e.Quants {
        let mark = q.Name == p.Quant.Name ? "*" : " "
        print("   \(mark) \(pad(q.Name, 10)) \(pad(size(q.Size), 9)) \(q.Files.count == 1 ? q.Files[0].Path : "\(q.Files.count) files")")
    }
    if let pr = e.Projector {
        print("  projector \(pr.Path)  \(size(pr.Size))")
    }
    print("  pulled    \(registry.IsPulled(p) ? "yes" : "no")")
}

func pull(_ ref: string, quiet: bool) async throws {
    let p = try registry.Find(ref)
    var h = hub.Hub()
    if !quiet {
        h.OnProgress = { pr in
            if pr.Done == pr.Size {
                print("  \(pr.File)  \(size(pr.Size))")
            } else {
                let pct = pr.Total > 0 ? pr.TotalDone * 100 / pr.Total : 0
                print("  \(pr.File)  \(size(pr.Done)) / \(size(pr.Size))  (\(pct)%)")
            }
        }
    }
    print("\(p.ID): \(size(p.Size)) from hf.co/\(p.Entry.Repo)")
    let got = try await registry.Pull(p, hub: h)
    print(got.Path.String())
}

func probe(_ picks: [registry.Pick]) async -> int32 {
    var bad = 0
    var n = 0
    for p in picks {
        do {
            for r in try await registry.Probe(p) {
                n += 1
                if !r.OK {
                    bad += 1
                }
                print("\(r.OK ? "ok  " : "FAIL") \(r.Status) \(pad(r.Magic, 4)) \(p.ID)  \(r.File)\(r.OK ? "" : "  (" + r.Why + ")")")
            }
        } catch {
            bad += 1
            print("FAIL \(p.ID): \(error)")
        }
    }
    print("\(n - bad) of \(n) files can be downloaded without a login")
    return bad == 0 ? 0 : 1
}

func main() async -> int32 {
    let args = CommandLine.arguments
    if args.count < 2 {
        return usage()
    }
    var refs: [string] = []
    var all = false
    var every = false
    var quiet = false
    for a in args.dropFirst(2) {
        if a == "--all" {
            all = true
        } else if a == "--every-quant" {
            every = true
        } else if a == "--quiet" {
            quiet = true
        } else if a.hasPrefix("-") {
            return usage()
        } else {
            refs.append(a)
        }
    }
    do {
        switch args[1] {
        case "list":
            list(refs.first)
        case "show":
            if refs.isEmpty { return usage() }
            for r in refs { try show(r) }
        case "pull":
            if refs.isEmpty { return usage() }
            for r in refs { try await pull(r, quiet: quiet) }
        case "probe":
            var picks: [registry.Pick] = []
            if all {
                for e in registry.Entries() {
                    for q in e.Quants where every || q.Name == e.Default {
                        picks.append(registry.Pick(Entry: e, Quant: q))
                    }
                }
            }
            for r in refs {
                picks.append(try registry.Find(r))
            }
            if picks.isEmpty { return usage() }
            return await probe(picks)
        default:
            return usage()
        }
    } catch {
        print("\(error)")
        return 1
    }
    return 0
}
