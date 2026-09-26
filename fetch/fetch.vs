// Package fetch opens a model by reference: a local file or directory, or
// a Hugging Face Hub reference whose files are fetched into the Hugging
// Face cache first. It is the only package under model that reaches the
// network, so a family, which imports model and never fetch, builds and
// runs offline. Task facades (llm.Load, tts.Load) call it.
//
//     let ck = try await fetch.Open("hf.co/HuggingFaceTB/SmolLM2-135M")
//     let ck = try await fetch.Open("hf.co/unsloth/Qwen3-0.6B-GGUF:Q4_K_M")
//     let ck = try await fetch.Open("./stories15M-q4_0.gguf")
//
// Choosing a format for a Hub repository is its job: the files are listed
// first (hub.Resolve moves no bytes), and only the chosen format's are
// downloaded.
package fetch

import (
    "fs"
    "model"
    "remote/hub"
)

/// FetchError is a reference that could not be opened, and why.
public enum FetchError: Error, CustomStringConvertible {
    case unsupported(string)

    public var description: string {
        switch self {
        case .unsupported(let why): return "fetch: " + why
        }
    }
}

/// Open is the checkpoint a reference names, its files fetched when it is
/// a Hub reference.
public func Open(_ ref: string, hub h: hub.Hub = hub.Hub()) async throws -> model.Checkpoint {
    if IsLocal(ref) {
        return try model.Open(fs.Path(ref))
    }
    let r = try hub.Ref.Parse(ref)
    var client = h
    let listed = try await client.Resolve(r)
    let choice = try Choose(listed.Files, ref: r)
    client.Include = choice.Include
    let snap = try await client.Download(r)
    if choice.Format == .gguf {
        let ggufs = snap.Files.filter { $0.Path.lowercased().hasSuffix(".gguf") }.sorted { $0.Path < $1.Path }
        if ggufs.count != 1 {
            throw FetchError.unsupported("\(r) is \(ggufs.count) GGUF files; a split GGUF is not read yet")
        }
        return try model.Open(snap.Path(ggufs[0].Path))
    }
    return try model.Open(snap.Dir)
}

/// IsLocal is whether a reference is a path on this machine rather than a
/// Hub reference: it starts with "/", "./", "../" or "~", or names
/// something that exists.
public func IsLocal(_ ref: string) -> bool {
    if ref.hasPrefix("/") || ref.hasPrefix("./") || ref.hasPrefix("../") || ref.hasPrefix("~") {
        return true
    }
    if ref.hasPrefix("hf.co/") || ref.hasPrefix("huggingface.co/") || ref.hasPrefix("https://") {
        return false
    }
    return (try? fs.Metadata(fs.Path(ref))) != nil
}

/// Choice is the format chosen for a Hub repository and the file patterns
/// that fetch it.
public struct Choice {
    public var Format: model.Format
    /// Include is hub.Hub's Include: empty to let the hub pick (a quant,
    /// a file the reference names).
    public var Include: [string]
}

/// Choose picks a Hub repository's format from its file list: GGUF for a
/// quant tag or a repository of nothing else; else safetensors, fetching
/// the top-level config, weights and tokenizer files only -- not a
/// PyTorch copy, ONNX, or Meta's original/ -- which is all a checkpoint
/// reads.
public func Choose(_ files: [hub.RepoFile], ref: hub.Ref) throws -> Choice {
    if !ref.File.isEmpty {
        let lower = ref.File.lowercased()
        if lower.hasSuffix(".gguf") {
            return Choice(Format: .gguf, Include: [])
        }
        return Choice(Format: .safetensors, Include: [])
    }
    if !ref.Tag.isEmpty {
        return Choice(Format: .gguf, Include: [])
    }
    var top: [string] = []
    for f in files where !f.Path.contains("/") {
        top.append(f.Path)
    }
    let safetensors = top.filter { $0.lowercased().hasSuffix(".safetensors") && !$0.hasPrefix("consolidated") }
    if !safetensors.isEmpty && top.contains("config.json") {
        return Choice(Format: .safetensors,
                      Include: ["/*.json", "/*.safetensors", "/tokenizer.model", "/*.jinja"])
    }
    if files.contains(where: { $0.Path.lowercased().hasSuffix(".gguf") }) {
        return Choice(Format: .gguf, Include: [])
    }
    throw FetchError.unsupported("\(ref) has neither safetensors with a config.json nor GGUF files")
}

/// Info is what a reference is, read without loading it.
public struct Info {
    public var Architecture: string
    public var Format: model.Format
    public var Tensors: int
    public var Bytes: int64
    public var Config: [(string, string)]
}

/// Inspect opens a reference -- fetching its files if it is on the Hub --
/// and says what it is: architecture, format, size and the config keys
/// asked for, read through aliases (a task's spellings for a GGUF), with
/// nothing put on a device.
public func Inspect(_ ref: string, keys: [string] = [], aliases: model.Aliases = model.Aliases()) async throws -> Info {
    let ck = try await Open(ref)
    ck.Aliases = aliases
    var bytes: int64 = 0
    for n in ck.Names {
        bytes += int64(ck.Info(n, gguf: n)?.Size ?? 0)
    }
    var cfg: [(string, string)] = []
    for key in keys {
        if let n = ck.Config.Int(key) {
            cfg.append((key, "\(n)"))
        } else if let x = ck.Config.Double(key) {
            cfg.append((key, "\(x)"))
        }
    }
    return Info(Architecture: ck.Architecture, Format: ck.Format, Tensors: ck.Names.count, Bytes: bytes, Config: cfg)
}
