// Package llama is the Llama family: Llama 1 and 2, and the models that
// share their architecture (general.architecture "llama" in GGUF) -- an
// embedding, blocks of RMSNorm, grouped-query attention with rotary
// position embeddings, RMSNorm and a SwiGLU MLP, each added back to what
// went in, then a final RMSNorm and the head to logits.
//
// This first cut runs a GGUF file one token at a time, greedily; the
// shared decoder kit, batching and sampling (model/decoder, model/generate)
// grow out of it as more families come.
package llama

import "fs"
import "gpu"
import "model/gguf"
import "nn"
import "tensor"
import "text/tokenizer"

/// LoadError is a file that is not a llama this can run, and says why.
public enum LoadError: Error {
    case unsupported(string)

    public var Message: string {
        switch self {
        case .unsupported(let why): return "llama: " + why
        }
    }
}

/// Config is a llama's shape, as its GGUF metadata gives it.
public struct Config {
    public var Layers: int
    public var Dim: int
    public var Hidden: int
    public var Heads: int
    public var KVHeads: int
    public var Vocab: int
    public var Context: int
    public var Eps: float32
    public var RopeBase: float32

    public var HeadDim: int { return Dim / Heads }

    /// Read is the config of a GGUF llama.
    public static func Read(_ f: gguf.File) throws -> Config {
        if f.Architecture != "llama" {
            throw LoadError.unsupported("general.architecture is '\(f.Architecture ?? "")', not 'llama'")
        }
        func need(_ key: string) throws -> int {
            guard let n = f.Integer("llama." + key) else {
                throw LoadError.unsupported("no llama.\(key)")
            }
            return n
        }
        let heads = try need("attention.head_count")
        let c = Config(
            Layers: try need("block_count"),
            Dim: try need("embedding_length"),
            Hidden: try need("feed_forward_length"),
            Heads: heads,
            KVHeads: f.Integer("llama.attention.head_count_kv") ?? heads,
            Vocab: f.Integer("llama.vocab_size") ?? f.Texts("tokenizer.ggml.tokens")?.count ?? 0,
            Context: try need("context_length"),
            Eps: float32(f.Number("llama.attention.layer_norm_rms_epsilon") ?? 1e-5),
            RopeBase: float32(f.Number("llama.rope.freq_base") ?? 10000))
        if let rope = f.Integer("llama.rope.dimension_count"), rope != c.HeadDim {
            throw LoadError.unsupported("rotary embeddings over \(rope) of a head's \(c.HeadDim) dimensions")
        }
        return c
    }
}

/// Vocabulary is a GGUF file's tokenizer.ggml.* data, for text/tokenizer.
public func Vocabulary(_ f: gguf.File) throws -> tokenizer.Vocabulary {
    if let model = f.Text("tokenizer.ggml.model"), model != "llama" {
        throw LoadError.unsupported("a '\(model)' tokenizer; this reads SentencePiece ('llama')")
    }
    guard let tokens = f.Texts("tokenizer.ggml.tokens"), let scores = f.Numbers("tokenizer.ggml.scores"),
          let types = f.Integers("tokenizer.ggml.token_type") else {
        throw LoadError.unsupported("no tokenizer.ggml tokens, scores and types")
    }
    return tokenizer.Vocabulary(
        tokens: tokens,
        scores: scores.map { float32($0) },
        kinds: types.map { tokenizer.Kind(rawValue: $0) ?? .Normal },
        bos: f.Integer("tokenizer.ggml.bos_token_id"),
        eos: f.Integer("tokenizer.ggml.eos_token_id"),
        unknown: f.Integer("tokenizer.ggml.unknown_token_id"),
        addBos: f.Flag("tokenizer.ggml.add_bos_token") ?? true,
        addEos: f.Flag("tokenizer.ggml.add_eos_token") ?? false,
        addSpacePrefix: f.Flag("tokenizer.ggml.add_space_prefix") ?? true)
}

/// Block is one transformer layer.
public final class Block {
    public let AttnNorm: nn.RMSNorm
    public let Attention: nn.Attention
    public let FFNNorm: nn.RMSNorm
    public let MLP: nn.GatedMLP

    init(attnNorm: nn.RMSNorm, attention: nn.Attention, ffnNorm: nn.RMSNorm, mlp: nn.GatedMLP) {
        self.AttnNorm = attnNorm
        self.Attention = attention
        self.FFNNorm = ffnNorm
        self.MLP = mlp
    }
}

/// Model is a llama on a device, with a KV cache for one sequence.
public final class Model {
    public let Config: Config
    public let Tokenizer: tokenizer.SentencePiece
    public let Device: gpu.Device
    public let Embed: nn.Embedding
    public let Blocks: [Block]
    public let Norm: nn.RMSNorm
    public let Output: nn.Linear
    let _caches: [nn.Cache]
    let _x: gpu.Buffer<float32>
    let _h: gpu.Buffer<float32>
    let _logits: gpu.Buffer<float32>

    init(_ config: Config, _ tok: tokenizer.SentencePiece, _ d: gpu.Device, _ embed: nn.Embedding, _ blocks: [Block],
         _ norm: nn.RMSNorm, _ output: nn.Linear) throws {
        self.Config = config
        self.Tokenizer = tok
        self.Device = d
        self.Embed = embed
        self.Blocks = blocks
        self.Norm = norm
        self.Output = output
        var caches: [nn.Cache] = []
        for _ in 0..<config.Layers {
            caches.append(try nn.Cache(on: d, kvHeads: config.KVHeads, headDim: config.HeadDim, capacity: config.Context))
        }
        self._caches = caches
        self._x = try d.CreateBuffer(of: float32.self, count: config.Dim)
        self._h = try d.CreateBuffer(of: float32.self, count: config.Dim)
        self._logits = try d.CreateBuffer(of: float32.self, count: output.Out)
    }

    /// Load reads the GGUF llama at path onto device d.
    public static func Load(_ path: fs.Path, on d: gpu.Device) async throws -> Model {
        let f = try gguf.Open(path)
        let c = try llama.Config.Read(f)
        func weight(_ name: string) async throws -> tensor.Tensor {
            guard let t = f.Tensor(name) else {
                throw LoadError.unsupported("no tensor \(name)")
            }
            let type: tensor.DType
            switch t.Type {
            case .F32: type = .F32
            case .Q4_0: type = .Q4_0
            case .Q8_0: type = .Q8_0
            default: throw LoadError.unsupported("\(name) is \(t.Type.Name), which this does not run yet")
            }
            // GGUF lists dimensions innermost first; a tensor, outermost.
            return try await tensor.Tensor.FromBytes(f.Bytes(t), shape: t.Shape.reversed(), dtype: type, on: d)
        }
        let embed = try nn.Embedding(try await weight("token_embd.weight"))
        var blocks: [Block] = []
        for i in 0..<c.Layers {
            let p = "blk.\(i)."
            let attention = try nn.Attention(
                q: try nn.Linear(try await weight(p + "attn_q.weight")),
                k: try nn.Linear(try await weight(p + "attn_k.weight")),
                v: try nn.Linear(try await weight(p + "attn_v.weight")),
                o: try nn.Linear(try await weight(p + "attn_output.weight")),
                heads: c.Heads, kvHeads: c.KVHeads, ropeBase: c.RopeBase)
            let mlp = try nn.GatedMLP(
                gate: try nn.Linear(try await weight(p + "ffn_gate.weight")),
                up: try nn.Linear(try await weight(p + "ffn_up.weight")),
                down: try nn.Linear(try await weight(p + "ffn_down.weight")),
                activation: .SiLU)
            blocks.append(Block(attnNorm: nn.RMSNorm(try await weight(p + "attn_norm.weight"), eps: c.Eps),
                                attention: attention,
                                ffnNorm: nn.RMSNorm(try await weight(p + "ffn_norm.weight"), eps: c.Eps),
                                mlp: mlp))
        }
        let norm = nn.RMSNorm(try await weight("output_norm.weight"), eps: c.Eps)
        // Without an output.weight the head is the embedding, tied.
        let output = f.Tensor("output.weight") != nil ? try nn.Linear(try await weight("output.weight")) : try nn.Linear(embed.Weight)
        let tok = tokenizer.SentencePiece(try Vocabulary(f))
        return try Model(c, tok, d, embed, blocks, norm, output)
    }

    /// Forward runs token at position through the model, adding it to the
    /// cache, and is its logits over the vocabulary.
    public func Forward(_ token: int, position: int) async throws -> gpu.Buffer<float32> {
        try await Embed.Lookup(token, into: _x)
        for i in 0..<Blocks.count {
            let b = Blocks[i]
            try await b.AttnNorm.Forward(_x, into: _h)
            try await b.Attention.Forward(_h, position: position, cache: _caches[i], into: _h)
            try await tensor.Add(_x, _h, into: _x)
            try await b.FFNNorm.Forward(_x, into: _h)
            try await b.MLP.Forward(_h, into: _h)
            try await tensor.Add(_x, _h, into: _x)
        }
        try await Norm.Forward(_x, into: _h)
        try await Output.Forward(_h, into: _logits)
        return _logits
    }

    /// Generate continues prompt greedily for at most n tokens, stopping at
    /// the end-of-sequence token, and is the tokens it made.
    public func Generate(_ prompt: string, tokens n: int) async throws -> [int] {
        let ids = Tokenizer.Encode(prompt)
        if ids.count + n > Config.Context {
            throw LoadError.unsupported("\(ids.count) prompt tokens and \(n) more are past the context of \(Config.Context)")
        }
        var logits: [float32] = []
        var position = 0
        for id in ids {
            logits = try await Forward(id, position: position).Download()
            position += 1
        }
        var out: [int] = []
        while out.count < n {
            let next = Argmax(logits)
            out.append(next)
            if next == Tokenizer.Vocab.Eos || out.count == n {
                break
            }
            logits = try await Forward(next, position: position).Download()
            position += 1
        }
        return out
    }
}

/// Argmax is the index of the largest value, the first of equals.
public func Argmax(_ xs: [float32]) -> int {
    var best = 0
    for i in 1..<max(1, xs.count) where xs[i] > xs[best] {
        best = i
    }
    return best
}
