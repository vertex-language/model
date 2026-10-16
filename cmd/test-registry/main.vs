// test-registry checks the list offline: every entry well formed and
// pinned, names unique, one latest per name, and every reference form
// Find takes resolving to what it should.
//
//     vsc run test-registry
package main

import "model/registry"

var failures = 0

func fail(_ what: string) {
    failures += 1
    if failures <= 40 {
        print("FAIL " + what)
    }
}

func isHex(_ s: string, _ n: int) -> bool {
    if s.count != n {
        return false
    }
    for c in s.utf8 {
        let digit = c >= 0x30 && c <= 0x39
        let lower = c >= 0x61 && c <= 0x66
        if !digit && !lower {
            return false
        }
    }
    return true
}

func main() -> int32 {
    let all = registry.Entries()
    var ids: [string: bool] = [:]
    var latest: [string: int] = [:]
    var tasks: [string: int] = [:]
    var files = 0
    var bytes: int64 = 0
    for e in all {
        if e.Name.isEmpty || e.Tag.isEmpty || e.Name != e.Name.lowercased() || e.Tag != e.Tag.lowercased() {
            fail("\(e.ID): a name and a tag, lowercase")
        }
        if e.Name.contains(":") || e.Name.contains(" ") || e.Tag.contains(":") || e.Tag.contains(" ") {
            fail("\(e.ID): no ':' or ' ' inside a name or tag")
        }
        if ids[e.ID] != nil {
            fail("\(e.ID): listed twice")
        }
        ids[e.ID] = true
        if e.Latest {
            latest[e.Name] = (latest[e.Name] ?? 0) + 1
        }
        tasks[e.Task] = (tasks[e.Task] ?? 0) + 1
        if !isHex(e.Revision, 40) {
            fail("\(e.ID): revision \(e.Revision) is not a commit")
        }
        if !e.Repo.contains("/") || !e.Source.contains("/") {
            fail("\(e.ID): repo and source are org/name")
        }
        if e.Quants.isEmpty {
            fail("\(e.ID): no quants")
        }
        if e.QuantNamed(e.Default) == nil {
            fail("\(e.ID): default quant \(e.Default) is not one of its quants")
        }
        var all = e.Quants.flatMap { $0.Files }
        if let p = e.Projector {
            all.append(p)
        }
        for q in e.Quants where q.Files.isEmpty {
            fail("\(e.ID): quant \(q.Name) has no files")
        }
        for f in all {
            files += 1
            bytes += f.Size
            if f.Size <= 0 {
                fail("\(e.ID): \(f.Path) has no size")
            }
            let ok = f.Hash.hasPrefix("git:") ? isHex(String(f.Hash.dropFirst(4)), 40) : isHex(f.Hash, 64)
            if !ok {
                fail("\(e.ID): \(f.Path) hash \(f.Hash) is neither a sha256 nor a git blob id")
            }
        }
    }
    for n in registry.Names() where latest[n] != 1 {
        fail("\(n): \(latest[n] ?? 0) latest tags, not 1")
    }

    // Every reference form, for every entry.
    for e in all {
        do {
            let p = try registry.Find(e.ID)
            if p.Entry.ID != e.ID || p.Quant.Name != e.Default {
                fail("Find(\(e.ID)) is \(p.ID)")
            }
            for q in e.Quants where q.Name != "files" {
                let ref = e.ID + "-" + q.Name
                let p = try registry.Find(ref.uppercased())
                if p.Entry.ID != e.ID || p.Quant.Name != q.Name {
                    fail("Find(\(ref)) is \(p.ID)")
                }
            }
            if e.Latest {
                for ref in [e.Name, e.Name + ":latest"] {
                    let p = try registry.Find(ref)
                    if p.Entry.ID != e.ID {
                        fail("Find(\(ref)) is \(p.ID), not \(e.ID)")
                    }
                }
            }
            let r = registry.Ref(p)
            if r.Repo != e.Repo || r.Revision != e.Revision {
                fail("Ref(\(e.ID)) is \(r)")
            }
        } catch {
            fail("Find(\(e.ID)): \(error)")
        }
    }
    // Names not on the list are errors, never guesses.
    for ref in ["no-such-model", "no-such-model:7b"] {
        if let p = try? registry.Find(ref) {
            fail("Find(\(ref)) is \(p.ID), not an error")
        }
    }
    if let first = all.first {
        if let p = try? registry.Find(first.Name + ":no-such-tag") {
            fail("Find(\(first.Name):no-such-tag) is \(p.ID), not an error")
        }
    }

    var summary: [string] = []
    for (t, n) in tasks.sorted(by: { $0.key < $1.key }) {
        summary.append("\(t) \(n)")
    }
    print("\(all.count) models (\(summary.joined(separator: ", "))), \(registry.Names().count) names, \(files) files, \(bytes / 1_000_000_000) GB")
    if failures > 0 {
        print("\(failures) failures")
        return 1
    }
    print("ok")
    return 0
}
