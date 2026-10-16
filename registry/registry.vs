// Package registry is models by name -- "qwen3", "qwen3:8b",
// "qwen3:8b-q8_0" -- each pinned to one repository on the Hugging Face
// Hub at one commit, with the size and hash of every file it is made of.
//
//     let pick = try registry.Find("qwen3:8b")
//     let model = try await registry.Pull("qwen3:8b")
//     print(model.Path)                // …/snapshots/<commit>/Qwen3-8B-Q4_K_M.gguf
//
// The list is table.vs. Being on it is a promise that the files download
// without a login, not that a family runs them: the list is the ladder of
// known models that support is measured against, one model at a time.
package registry

import (
    "fs"
    "net/http"
    "net/url"
    "remote/hub"
)

/// File is one file of a model: its path in the repository, its size,
/// and its hash.
public struct File {
    public let Path: string
    public let Size: int64
    /// The sha256 of a large (LFS) file; for a small one, "git:" and its
    /// git blob id, as the Hub lists them.
    public let Hash: string

    /// HubFile is the file as remote/hub knows it, for its cache.
    public var HubFile: hub.RepoFile {
        if self.Hash.hasPrefix("git:") {
            return hub.RepoFile(path: self.Path, size: self.Size, blobId: String(self.Hash.dropFirst(4)))
        }
        return hub.RepoFile(path: self.Path, size: self.Size, sha256: self.Hash)
    }
}

/// Quant is one quantization of a model, named as llama.cpp names it,
/// lowercased ("q4_k_m", "q8_0", "bf16"), and its files: one, or the
/// parts of a split GGUF. A model that comes as its own files rather than
/// a GGUF (Kokoro) has one Quant, named "files".
public struct Quant {
    public let Name: string
    public let Files: [File]

    /// The bytes of all its files.
    public var Size: int64 {
        var n: int64 = 0
        for f in self.Files {
            n += f.Size
        }
        return n
    }
}

/// Entry is one model on the list.
public struct Entry {
    /// The name, generation included: "qwen3", "llama3.2", "gemma3".
    public let Name: string
    /// The size and variant: "8b", "30b-a3b", "4b-2507".
    public let Tag: string
    /// The task repository it belongs to: "llm", "embed", "tts", "asr",
    /// "diffusion", "vision", "forecast", or "other".
    public let Task: string
    /// The repository the files come from: "unsloth/Qwen3-8B-GGUF".
    public let Repo: string
    /// The commit they are pinned to.
    public let Revision: string
    /// The original model the files are a conversion of: "Qwen/Qwen3-8B".
    public let Source: string
    /// GGUF's general.architecture ("qwen3"), or "-" when unknown.
    public let Arch: string
    public let License: string
    /// The quant a tag without one means: "q4_k_m" where there is one.
    public let Default: string
    /// Whether this is the tag the bare name means.
    public let Latest: bool
    public let Quants: [Quant]
    /// A vision-language model's projector (mmproj), fetched with it.
    public let Projector: File?

    /// ID is "name:tag".
    public var ID: string {
        return self.Name + ":" + self.Tag
    }

    /// QuantNamed is the quant of that name, or nil.
    public func QuantNamed(_ name: string) -> Quant? {
        for q in self.Quants where q.Name == name {
            return q
        }
        return nil
    }
}

/// Pick is what a reference names: an entry, and one of its quants.
public struct Pick {
    public let Entry: Entry
    public let Quant: Quant

    public init(Entry: Entry, Quant: Quant) {
        self.Entry = Entry
        self.Quant = Quant
    }

    /// ID is "name:tag-quant", or "name:tag" for a model of its own files.
    public var ID: string {
        if self.Quant.Name == "files" {
            return self.Entry.ID
        }
        return self.Entry.ID + "-" + self.Quant.Name
    }

    /// Files are what Pull fetches: the quant's, and the projector.
    public var Files: [File] {
        var out = self.Quant.Files
        if let p = self.Entry.Projector {
            out.append(p)
        }
        return out
    }

    /// The bytes Pull fetches.
    public var Size: int64 {
        var n: int64 = 0
        for f in self.Files {
            n += f.Size
        }
        return n
    }
}

/// RegistryError is a reference that names nothing on the list, or a
/// repository that no longer holds what the list pinned.
public enum RegistryError: Error, CustomStringConvertible {
    /// A name not on the list, and the names that look like it.
    case unknownName(string, [string])
    /// A known name with a tag or quant it does not have, and its tags.
    case unknownTag(string, [string])
    /// A pinned file that the repository lists differently, and how.
    case changed(string, string)
    /// A model the Hub will not give without a login, and why.
    case needsLogin(string, string)
    /// The Hub is turning this address away for now.
    case rateLimited(string)

    public var description: string {
        switch self {
        case .unknownName(let n, let near):
            if near.isEmpty {
                return "registry: no model is named \(n)"
            }
            return "registry: no model is named \(n); did you mean \(near.joined(separator: ", "))?"
        case .unknownTag(let ref, let tags):
            return "registry: \(ref) is not on the list; there is \(tags.joined(separator: ", "))"
        case .changed(let file, let why):
            return "registry: \(file) \(why); the list needs a new pin"
        case .needsLogin(let id, let why):
            return "registry: \(id): \(why). Set HF_TOKEN to a read token (huggingface.co/settings/tokens), or run `hf auth login`"
        case .rateLimited(let id):
            return "registry: \(id): the Hub is rate-limiting this address (without a login, 500 API calls in five minutes). Wait a few minutes and pull again, or set HF_TOKEN for a higher limit"
        }
    }
}

// table is tableText (table.vs) read, once, the first time the list is
// asked for.
let table: [Entry] = parseTable(tableText)

/// Entries is the whole list, by task, name and tag.
public func Entries() -> [Entry] {
    return table
}

// parseTable reads the list's text (table.vs says its format). A line it
// cannot read is skipped; test-registry counts what it read.
func parseTable(_ text: string) -> [Entry] {
    var out: [Entry] = []
    var head: [string] = []
    var files: [(string, File)] = []
    for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
        let w = line.split(separator: " ", omittingEmptySubsequences: true).map { String($0) }
        if line.hasPrefix(" ") {
            if w.count == 4, let size = int64(w[2]) {
                files.append((w[0], File(Path: w[1], Size: size, Hash: w[3])))
            }
            continue
        }
        if let e = entry(head, files) {
            out.append(e)
        }
        head = w
        files = []
    }
    if let e = entry(head, files) {
        out.append(e)
    }
    return out
}

// entry is an entry line's words and its files, or nil where they are
// not one: name:tag task repo@commit source arch license default [latest].
func entry(_ w: [string], _ files: [(string, File)]) -> Entry? {
    if w.count < 7 || files.isEmpty {
        return nil
    }
    let nameTag = w[0].split(separator: ":", maxSplits: 1).map { String($0) }
    let repoRev = w[2].split(separator: "@", maxSplits: 1).map { String($0) }
    if nameTag.count != 2 || repoRev.count != 2 {
        return nil
    }
    var order: [string] = []
    var projector: File? = nil
    for (q, f) in files {
        if q == "mmproj" {
            projector = f
        } else if !order.contains(q) {
            order.append(q)
        }
    }
    var quants: [Quant] = []
    for q in order {
        var fs: [File] = []
        for (name, f) in files where name == q {
            fs.append(f)
        }
        quants.append(Quant(Name: q, Files: fs))
    }
    return Entry(Name: nameTag[0], Tag: nameTag[1], Task: w[1], Repo: repoRev[0], Revision: repoRev[1],
                 Source: w[3], Arch: w[4], License: w[5], Default: w[6] == "-" ? "files" : w[6],
                 Latest: w.count > 7 && w[7] == "latest", Quants: quants, Projector: projector)
}

/// Names are the list's names, each once, in order.
public func Names() -> [string] {
    var out: [string] = []
    var last = ""
    for e in table where e.Name != last {
        if !out.contains(e.Name) {
            out.append(e.Name)
        }
        last = e.Name
    }
    return out
}

/// Find is what a reference names:
///
///     qwen3              the latest tag, at its default quant
///     qwen3:latest       the same
///     qwen3:8b           that tag, at its default quant
///     qwen3:8b-q8_0      that tag and quant
///
/// Names and tags are matched without case.
public func Find(_ ref: string) throws -> Pick {
    let text = ref.lowercased()
    var name = text
    var tag = ""
    if let colon = text.firstIndex(of: ":") {
        name = String(text[text.startIndex..<colon])
        tag = String(text[text.index(after: colon)...])
    }
    var of: [Entry] = []
    for e in table where e.Name == name {
        of.append(e)
    }
    if of.isEmpty {
        throw RegistryError.unknownName(name, Suggest(name))
    }
    if tag.isEmpty || tag == "latest" {
        for e in of where e.Latest {
            return try pick(e, e.Default, ref)
        }
        return try pick(of[0], of[0].Default, ref)
    }
    for e in of where e.Tag == tag {
        return try pick(e, e.Default, ref)
    }
    // "8b-q8_0": the longest tag that is a prefix, and a quant after it.
    var best: Entry? = nil
    for e in of where tag.hasPrefix(e.Tag + "-") {
        let q = String(tag.dropFirst(e.Tag.count + 1))
        if e.QuantNamed(q) != nil && (best == nil || e.Tag.count > best!.Tag.count) {
            best = e
        }
    }
    if let e = best {
        return try pick(e, String(tag.dropFirst(e.Tag.count + 1)), ref)
    }
    if tag.hasPrefix("latest-") {
        for e in of where e.Latest {
            return try pick(e, String(tag.dropFirst(7)), ref)
        }
    }
    var tags: [string] = []
    for e in of {
        tags.append(e.ID)
    }
    throw RegistryError.unknownTag(ref, tags)
}

func pick(_ e: Entry, _ quant: string, _ ref: string) throws -> Pick {
    let want = quant.isEmpty || quant == "-" ? "files" : quant
    guard let q = e.QuantNamed(want) else {
        var names: [string] = []
        for q in e.Quants {
            names.append(e.ID + "-" + q.Name)
        }
        throw RegistryError.unknownTag(ref, names)
    }
    return Pick(Entry: e, Quant: q)
}

/// Suggest is up to eight names that look like name: containing it,
/// contained in it, or sharing its first three letters.
public func Suggest(_ name: string) -> [string] {
    let head = String(name.prefix(3))
    var out: [string] = []
    for n in Names() {
        if out.count == 8 {
            break
        }
        if n.contains(name) || (name.count > 2 && name.contains(n)) || (head.count == 3 && n.hasPrefix(head)) {
            out.append(n)
        }
    }
    return out
}

/// Ref is the hub reference a pick downloads from: its repository at its
/// pinned commit.
public func Ref(_ p: Pick) -> hub.Ref {
    return hub.Ref(p.Entry.Repo, revision: p.Entry.Revision)
}

/// Pulled is a pick whose files are in the Hugging Face cache.
public struct Pulled {
    public let Pick: Pick
    /// The snapshot directory the files are linked into.
    public let Dir: fs.Path

    /// Path is what to open: the GGUF file (a split one's first part), or
    /// the snapshot directory for a model of its own files.
    public var Path: fs.Path {
        if self.Pick.Quant.Name == "files" || self.Pick.Quant.Files.isEmpty {
            return self.Dir
        }
        return self.Dir / self.Pick.Quant.Files[0].Path
    }

    /// The projector's path, for a vision-language model.
    public var Projector: fs.Path? {
        guard let p = self.Pick.Entry.Projector else {
            return nil
        }
        return self.Dir / p.Path
    }
}

/// Pull downloads what a reference names into the Hugging Face cache --
/// only what is missing -- and is where it is.
public func Pull(_ ref: string, hub h: hub.Hub = hub.Hub()) async throws -> Pulled {
    return try await Pull(try Find(ref), hub: h)
}

/// Pull downloads a pick's files at its pinned commit: one API call (the
/// Hub allows an anonymous address 500 in five minutes), then the files.
/// remote/hub checks each download against the hash the Hub lists, and
/// those are checked against the list's.
public func Pull(_ p: Pick, hub h: hub.Hub = hub.Hub()) async throws -> Pulled {
    var client = h
    client.Include = []
    for f in p.Files {
        client.Include.append("/" + f.Path)
    }
    do {
        let snap = try await client.Download(Ref(p))
        try Check(p, snap.Files)
        return Pulled(Pick: p, Dir: snap.Dir)
    } catch let e as hub.HubError {
        throw explain(p, e)
    }
}

// explain says what a hub error means for a model on the list.
func explain(_ p: Pick, _ e: hub.HubError) -> Error {
    switch e {
    case .gated(let r):
        return RegistryError.needsLogin(p.ID, "hf.co/\(r) is gated: accept its terms on the Hub")
    case .repoNotFound(let r):
        return RegistryError.needsLogin(p.ID, "hf.co/\(r) is gone, or private")
    case .revisionNotFound(let r, let rev):
        return RegistryError.changed(r, "no longer has commit \(rev)")
    case .status(let code, let msg):
        if code == 429 {
            return RegistryError.rateLimited(p.ID)
        }
        if code == 401 || code == 403 {
            return RegistryError.needsLogin(p.ID, "the Hub answered \(code) \(msg)")
        }
        return e
    default:
        return e
    }
}

/// Check is whether a repository's listed files hold what the list
/// pinned for a pick: every file there, at its size and hash.
public func Check(_ p: Pick, _ listed: [hub.RepoFile]) throws {
    for f in p.Files {
        var found: hub.RepoFile? = nil
        for l in listed where l.Path == f.Path {
            found = l
            break
        }
        guard let l = found else {
            throw RegistryError.changed(f.Path, "is not in \(p.Entry.Repo) at \(p.Entry.Revision)")
        }
        if l.Size != f.Size {
            throw RegistryError.changed(f.Path, "is \(l.Size) bytes, not \(f.Size)")
        }
        if l.Etag.lowercased() != f.HubFile.Etag.lowercased() {
            throw RegistryError.changed(f.Path, "has hash \(l.Etag), not \(f.HubFile.Etag)")
        }
    }
}

/// IsPulled is whether all of a pick's files are in the cache already.
public func IsPulled(_ p: Pick, cache: hub.Cache = hub.Cache.Default()) -> bool {
    for f in p.Files {
        if !cache.HasBlob(hub.RepoKind.model, p.Entry.Repo, f.HubFile) {
            return false
        }
    }
    return true
}

/// Probed is what asking for a file's first bytes found.
public struct Probed {
    public let File: string
    /// The final HTTP status: 206 or 200 when the bytes came.
    public let Status: int
    /// The first four bytes, as text where they are ("GGUF").
    public let Magic: string

    /// OK is whether the bytes came, and a GGUF file begins with "GGUF".
    public var OK: bool {
        return self.Why.isEmpty
    }

    /// Why is what went wrong, or "".
    public var Why: string {
        switch self.Status {
        case 200, 206:
            if self.File.lowercased().hasSuffix(".gguf") && self.Magic != "GGUF" {
                return "not a GGUF file"
            }
            return ""
        case 401, 403:
            return "needs a login (HF_TOKEN)"
        case 404:
            return "not found at the pinned commit"
        case 429:
            return "rate-limited: wait and try again"
        default:
            return "HTTP \(self.Status)"
        }
    }
}

/// Probe asks for the first four bytes of each of a pick's files as a
/// stranger would -- no token, following the Hub's redirect to its CDN --
/// and is what came back. It shows a file can be downloaded without a
/// login while moving a few bytes, not gigabytes.
public func Probe(_ p: Pick, endpoint: string = "https://huggingface.co") async throws -> [Probed] {
    var out: [Probed] = []
    for f in p.Files {
        var segs: [string] = []
        for s in f.Path.split(separator: "/") {
            segs.append(url.PathEscape(String(s)))
        }
        let address = "\(endpoint)/\(p.Entry.Repo)/resolve/\(p.Entry.Revision)/\(segs.joined(separator: "/"))"
        out.append(try await probe(f.Path, address))
    }
    return out
}

func probe(_ file: string, _ address: string) async throws -> Probed {
    let client = http.Client(timeoutMs: 30000)
    var u = try url.Parse(address)
    var hops = 0
    while true {
        var req = http.Request(method: "GET", url: u.RequestURI)
        req.Headers.Set("User-Agent", "vertex-registry/0.1")
        req.Headers.Set("Range", "bytes=0-3")
        var s = try await client.Open(req, url: u)
        let code = int(s.Response.StatusCode)
        if code == 301 || code == 302 || code == 303 || code == 307 || code == 308 {
            let loc = s.Response.Headers.Get("Location")
            s.Close()
            hops += 1
            guard let l = loc, hops <= 10 else {
                return Probed(File: file, Status: code, Magic: "")
            }
            u = try u.Resolve(l)
            continue
        }
        defer { s.Close() }
        if code != 200 && code != 206 {
            return Probed(File: file, Status: code, Magic: "")
        }
        var got: [uint8] = []
        var buf = [uint8](repeating: 0, count: 4)
        while got.count < 4 {
            let n = try await s.Read(into: &buf)
            if n == 0 {
                break
            }
            got.append(contentsOf: buf[0..<n])
        }
        var magic = ""
        for b in got.prefix(4) where b >= 0x20 && b < 0x7F {
            magic += String(UnicodeScalar(b))
        }
        return Probed(File: file, Status: code, Magic: magic)
    }
}
