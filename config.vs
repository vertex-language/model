package model

import (
    "encoding/json"
    "model/gguf"
)

/// Config is a checkpoint's configuration, asked for in Hugging Face's
/// vocabulary (config.json's keys) whatever the format. For a GGUF file a
/// key is translated through the checkpoint's Aliases (llm/arch's for a
/// decoder); a family's own keys name their GGUF spelling themselves:
///
///     c.Int("num_hidden_layers")                       // llama.block_count in a GGUF
///     c.Double("query_pre_attn_scalar", gguf: "attention.query_pre_attn_scalar")
///
/// A GGUF key without a "general." or "tokenizer." prefix is under the
/// architecture's: "block_count" is "llama.block_count".
public struct Config {
    let hf: json.Value?
    let file: gguf.File?
    let arch: string
    var keys: [string: string] = [:]

    init(hf: json.Value) {
        self.hf = hf
        self.file = nil
        self.arch = ""
    }

    init(gguf f: gguf.File) {
        self.hf = nil
        self.file = f
        self.arch = f.Architecture ?? ""
    }

    /// Raw is config.json itself, for what only it has (rope_scaling's
    /// object, a quantization_config); nil for a GGUF file.
    public var Raw: json.Value? { return hf }

    /// Has is whether the key is set.
    public func Has(_ key: string, gguf: string? = nil) -> bool {
        if let v = hf {
            if let x = v[key] {
                return !x.IsNull
            }
            return false
        }
        guard let k = ggufKey(key, gguf) else { return false }
        return file!.Value(k) != nil
    }

    public func Int(_ key: string, gguf: string? = nil) -> int? {
        if let v = hf {
            if let n = v[key]?.Int {
                return int(n)
            }
            return nil
        }
        guard let k = ggufKey(key, gguf) else { return nil }
        if let n = file!.Integer(k) {
            return n
        }
        // A GGUF has no vocab_size where its tokens say it.
        if key == "vocab_size" && gguf == nil {
            return file!.Texts("tokenizer.ggml.tokens")?.count
        }
        return nil
    }

    public func Double(_ key: string, gguf: string? = nil) -> float64? {
        if let v = hf {
            return v[key]?.Double
        }
        guard let k = ggufKey(key, gguf) else { return nil }
        return file!.Number(k) ?? file!.Integer(k).map { float64($0) }
    }

    public func Bool(_ key: string, gguf: string? = nil) -> bool? {
        if let v = hf {
            return v[key]?.Bool
        }
        guard let k = ggufKey(key, gguf) else { return nil }
        return file!.Flag(k)
    }

    public func String(_ key: string, gguf: string? = nil) -> string? {
        if let v = hf {
            return v[key]?.String
        }
        guard let k = ggufKey(key, gguf) else { return nil }
        return file!.Text(k)
    }

    /// Ints is an integer list: config.json's eos_token_id may be one.
    public func Ints(_ key: string, gguf: string? = nil) -> [int]? {
        if let v = hf {
            guard let x = v[key] else { return nil }
            if let n = x.Int {
                return [int(n)]
            }
            guard let a = x.Array else { return nil }
            var out: [int] = []
            for e in a {
                guard let n = e.Int else { return nil }
                out.append(int(n))
            }
            return out
        }
        guard let k = ggufKey(key, gguf) else { return nil }
        if let n = file!.Integer(k) {
            return [n]
        }
        return file!.Integers(k)
    }

    // ggufKey is the GGUF key an HF key is asked as.
    func ggufKey(_ key: string, _ override: string?) -> string? {
        guard let k = override ?? keys[key] else {
            return nil
        }
        if k.hasPrefix("general.") || k.hasPrefix("tokenizer.") {
            return k
        }
        return arch + "." + k
    }
}
