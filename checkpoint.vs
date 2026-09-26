// Package model is what every model shares, and no model: a model's files
// seen one way whatever their format -- a configuration, tensors by name, a
// tokenizer and a chat template. Open reads a Hugging Face snapshot
// directory (config.json and safetensors, sharded or not) or a GGUF file,
// offline; model/fetch gets a Hub reference's files first.
//
// A checkpoint answers in Hugging Face's vocabulary -- config keys, tensor
// names, shapes outermost first -- whatever the format, so a family is
// written once for all of them. How a task's names are spelled in a GGUF
// is the task's to say (Aliases): llm/arch gives llama.cpp's decoder
// tables. Architecture is config.json's architectures (or model_type), or
// a GGUF's general.architecture; a family decides whether it claims one.
//
// The formats are its subpackages: model/safetensors, model/gguf.
package model

import (
    "encoding/json"
    "fs"
    "gpu"
    "model/gguf"
    "model/safetensors"
    "model/torch"
    "tensor"
    "text/tokenizer"
)

/// Format is what a checkpoint's weights are stored as.
public enum Format: Equatable {
    case safetensors
    case gguf
    /// A PyTorch checkpoint (.pth, .pt, pytorch_model.bin), read through
    /// model/torch's restricted unpickler.
    case torch

    public var Name: string {
        switch self {
        case .safetensors: return "safetensors"
        case .gguf: return "gguf"
        case .torch: return "torch"
        }
    }
}

/// OpenError is a checkpoint that could not be opened or read, and why.
public enum OpenError: Error, CustomStringConvertible {
    case unsupported(string)

    public var description: string {
        switch self {
        case .unsupported(let why): return "model: " + why
        }
    }
}

/// TensorInfo is what a checkpoint says of a tensor, before any byte of it
/// is read: its name in the file, shape (outermost first), dtype, size.
public struct TensorInfo {
    /// Name is the tensor's name in its file: "blk.0.attn_q.weight" in a GGUF.
    public let Name: string
    public let Shape: [int]
    /// DType is the element type as tensor has it, or nil where tensor has
    /// none for it yet.
    public let DType: tensor.DType?
    /// SourceType is the file's own name for the dtype: "BF16", "q4_K".
    public let SourceType: string
    public let Size: int
    let file: int
    let offset: int
}

/// Aliases are a task's spellings of its Hugging Face names in a GGUF
/// file: config.json keys as GGUF keys (under the architecture unless they
/// start "general." or "tokenizer."), and tensor names. llama.cpp's
/// converter renames both, and each task's GGUFs follow their own table.
public struct Aliases {
    public var ConfigKeys: [string: string]
    public var TensorName: (string) -> string?

    public init(configKeys: [string: string] = [:], tensorName: @escaping (string) -> string? = { _ in nil }) {
        self.ConfigKeys = configKeys
        self.TensorName = tensorName
    }
}

/// Checkpoint is a model's files, opened.
public final class Checkpoint {
    public let Format: Format
    /// Path is what was opened: a directory, or a file.
    public let Path: fs.Path
    /// Architecture is the family's name for it: "LlamaForCausalLM", or
    /// a GGUF's "llama".
    public let Architecture: string
    /// Aliases are how the task reading this checkpoint spells its
    /// Hugging Face names in a GGUF; a family sets its task's before it
    /// reads (llm/arch's DecoderAliases). Empty, only names a GGUF has
    /// answer.
    public var Aliases: model.Aliases = model.Aliases()
    let _config: Config
    let _dir: fs.Path?
    let _st: [safetensors.File]
    let _pt: [torch.File]
    let _gguf: gguf.File?
    let _info: [string: TensorInfo]
    let _names: [string]
    var _device: gpu.Device? = nil
    var _wraps: [gpu.Buffer<uint8>?] = []

    init(format: Format, path: fs.Path, dir: fs.Path?, architecture: string, config: Config,
         st: [safetensors.File], pt: [torch.File] = [], gguf file: gguf.File?, info: [string: TensorInfo], names: [string]) {
        self.Format = format
        self.Path = path
        self._dir = dir
        self.Architecture = architecture
        self._config = config
        self._st = st
        self._pt = pt
        self._gguf = file
        self._info = info
        self._names = names
    }

    /// Config is the checkpoint's configuration, asked in Hugging Face's
    /// keys, through Aliases for a GGUF.
    public var Config: Config {
        var c = _config
        c.keys = Aliases.ConfigKeys
        return c
    }

    /// Dir is the directory the checkpoint's files are in: where a family
    /// finds what else it reads (Kokoro's voices/).
    public var Dir: fs.Path? { return _dir }

    /// Names are the tensors' names in their files, in file order.
    public var Names: [string] { return _names }

    /// GGUF is the file, for a GGUF checkpoint: its metadata beyond the
    /// config view.
    public var GGUF: gguf.File? { return _gguf }

    /// Info is the tensor a Hugging Face name means -- for a GGUF through
    /// Aliases, or the GGUF name given -- or nil where there is none.
    public func Info(_ name: string, gguf: string? = nil) -> TensorInfo? {
        if Format == .safetensors {
            return _info[name]
        }
        if let g = gguf {
            return _info[g]
        }
        if let g = Aliases.TensorName(name), let t = _info[g] {
            return t
        }
        return _info[name]
    }

    /// Has is whether the tensor is there.
    public func Has(_ name: string, gguf: string? = nil) -> bool {
        return Info(name, gguf: gguf) != nil
    }

    /// Tensor is the named tensor on device d. Where the device can read
    /// the file's mapping in place (unified memory), the tensor is a slice
    /// of it and nothing is copied; else its bytes are uploaded.
    public func Tensor(_ name: string, gguf: string? = nil, on d: gpu.Device) async throws -> tensor.Tensor {
        guard let t = Info(name, gguf: gguf) else {
            throw OpenError.unsupported("\(Path.String()) has no tensor \(name)")
        }
        guard let dt = t.DType else {
            throw OpenError.unsupported("\(t.Name) is \(t.SourceType), which tensor does not hold yet")
        }
        if let all = try wrap(t.file, on: d), t.offset % 16 == 0 {
            return try tensor.Tensor(shape: t.Shape, dtype: dt, storage: all.Slice(from: t.offset, count: t.Size))
        }
        return try await tensor.Tensor.FromBytes(bytes(t), shape: t.Shape, dtype: dt, on: d)
    }

    /// Float32 is the named tensor as float32 on d, decoded if it is not:
    /// what a norm's weight is.
    public func Float32(_ name: string, gguf: string? = nil, on d: gpu.Device) async throws -> tensor.Tensor {
        return try await Tensor(name, gguf: gguf, on: d).ToF32()
    }

    /// Copy is a tensor's bytes as its file holds them, copied out.
    public func Copy(_ t: TensorInfo) -> [uint8] {
        var out = [uint8](repeating: 0, count: t.Size)
        let p = bytes(t)
        out.withUnsafeMutableBufferPointer { dst in
            var i = 0
            while i < t.Size {
                dst[i] = p[i]
                i += 1
            }
        }
        return out
    }

    // bytes is where a tensor's bytes lie in its file's mapping.
    func bytes(_ t: TensorInfo) -> UnsafePointer<uint8> {
        if let f = _gguf {
            return f.Mapping.Bytes! + t.offset
        }
        if Format == .torch {
            return _pt[t.file].Mapping.Bytes! + t.offset
        }
        return _st[t.file].Mapping.Bytes! + t.offset
    }

    // wrap is file i's whole mapping as one buffer d reads in place, made
    // once a device; nil where d will not take it.
    func wrap(_ i: int, on d: gpu.Device) throws -> gpu.Buffer<uint8>? {
        if _device == nil || !(_device! === d) {
            _device = d
            let n = _gguf != nil ? 1 : (Format == .torch ? _pt.count : _st.count)
            _wraps = []
            for j in 0..<n {
                _wraps.append(nil)
                if let f = _gguf {
                    _wraps[j] = try? d.Wrap(UnsafeMutableRawPointer(mutating: f.Mapping.Bytes!), bytes: f.Mapping.Count, keeping: f)
                } else if Format == .torch {
                    let f = _pt[j]
                    _wraps[j] = try? d.Wrap(UnsafeMutableRawPointer(mutating: f.Mapping.Bytes!), bytes: f.Mapping.Count, keeping: f)
                } else {
                    let f = _st[j]
                    _wraps[j] = try? d.Wrap(UnsafeMutableRawPointer(mutating: f.Mapping.Bytes!), bytes: f.Mapping.Count, keeping: f)
                }
            }
        }
        return _wraps[i]
    }

    /// ChatTemplate is the model's Jinja chat template: a GGUF's
    /// tokenizer.chat_template, or chat_template.jinja, or
    /// tokenizer_config.json's chat_template (the default one, where it
    /// holds several).
    public var ChatTemplate: string? {
        if let f = _gguf {
            return f.Text("tokenizer.chat_template")
        }
        guard let dir = _dir else { return nil }
        if let t = try? fs.ReadText(dir / "chat_template.jinja") {
            return t
        }
        guard let tc = readJSON(dir / "tokenizer_config.json"), let ct = tc["chat_template"] else {
            return nil
        }
        if let s = ct.String {
            return s
        }
        for e in ct.Array ?? [] where e["name"]?.String == "default" {
            return e["template"]?.String
        }
        return nil
    }

    /// Tokenizer is the model's tokenizer, from wherever its files keep it:
    /// a GGUF's tokenizer.ggml.* keys (SentencePiece or byte-level BPE);
    /// else a SentencePiece tokenizer.model, with tokenizer_config.json's say
    /// on BOS and EOS; else a byte-level BPE tokenizer.json.
    public func Tokenizer() throws -> tokenizer.Tokenizer {
        if let f = _gguf {
            return try ggufTokenizer(f)
        }
        guard let dir = _dir else {
            throw OpenError.unsupported("no tokenizer beside \(Path.String())")
        }
        let config = try? fs.ReadFile(dir / "tokenizer_config.json")
        if let model = try? fs.ReadFile(dir / "tokenizer.model") {
            var v = try tokenizer.ReadSentencePieceModel(model)
            if let c = config, let tc = try? json.Parse(bytes: c) {
                if let b = tc["add_bos_token"]?.Bool { v.AddBos = b }
                if let e = tc["add_eos_token"]?.Bool { v.AddEos = e }
            }
            return tokenizer.Tokenizer.SentencePiece(v)
        }
        if let j = try? fs.ReadFile(dir / "tokenizer.json") {
            return try tokenizer.ReadTokenizerJSON(j, config: config)
        }
        throw OpenError.unsupported("\(dir.String()) has no tokenizer.model or tokenizer.json")
    }
}

// ggufTokenizer is a GGUF's tokenizer.ggml.* data: SentencePiece ("llama")
// or byte-level BPE ("gpt2", its pre-tokenizer named by tokenizer.ggml.pre).
func ggufTokenizer(_ f: gguf.File) throws -> tokenizer.Tokenizer {
    let model = f.Text("tokenizer.ggml.model") ?? "llama"
    guard let tokens = f.Texts("tokenizer.ggml.tokens"), let types = f.Integers("tokenizer.ggml.token_type") else {
        throw OpenError.unsupported("no tokenizer.ggml tokens and types")
    }
    let kinds = types.map { tokenizer.Kind(rawValue: $0) ?? .Normal }
    switch model {
    case "llama":
        guard let scores = f.Numbers("tokenizer.ggml.scores") else {
            throw OpenError.unsupported("a SentencePiece vocabulary without scores")
        }
        return tokenizer.Tokenizer.SentencePiece(tokenizer.Vocabulary(
            tokens: tokens,
            scores: scores.map { float32($0) },
            kinds: kinds,
            bos: f.Integer("tokenizer.ggml.bos_token_id"),
            eos: f.Integer("tokenizer.ggml.eos_token_id"),
            unknown: f.Integer("tokenizer.ggml.unknown_token_id"),
            addBos: f.Flag("tokenizer.ggml.add_bos_token") ?? true,
            addEos: f.Flag("tokenizer.ggml.add_eos_token") ?? false,
            addSpacePrefix: f.Flag("tokenizer.ggml.add_space_prefix") ?? true))
    case "gpt2":
        let pre = f.Text("tokenizer.ggml.pre") ?? "default"
        guard let split = tokenizer.Split.Named(pre) else {
            throw OpenError.unsupported("a BPE pre-tokenizer '\(pre)' not implemented yet")
        }
        guard let merges = f.Texts("tokenizer.ggml.merges") else {
            throw OpenError.unsupported("a BPE vocabulary without merges")
        }
        var v = tokenizer.Vocabulary(
            tokens: tokens, scores: [], kinds: kinds,
            bos: f.Integer("tokenizer.ggml.bos_token_id"),
            eos: f.Integer("tokenizer.ggml.eos_token_id"),
            unknown: f.Integer("tokenizer.ggml.unknown_token_id"),
            addBos: f.Flag("tokenizer.ggml.add_bos_token") ?? false,
            addEos: f.Flag("tokenizer.ggml.add_eos_token") ?? false,
            addSpacePrefix: false)
        v.Merges = merges
        return tokenizer.Tokenizer.BPE(v, split: split)
    default:
        throw OpenError.unsupported("a '\(model)' tokenizer; SentencePiece and byte-level BPE are read so far")
    }
}

func readJSON(_ p: fs.Path) -> json.Value? {
    guard let b = try? fs.ReadFile(p) else { return nil }
    return try? json.Parse(bytes: b)
}

/// Open opens a checkpoint: a GGUF file; a safetensors file (with the
/// config.json beside it); or a directory holding config.json and
/// safetensors -- one file, shards with model.safetensors.index.json, or
/// shards without -- or else one GGUF file.
public func Open(_ path: fs.Path) throws -> Checkpoint {
    let meta = try fs.Metadata(path)
    if !meta.IsDir() {
        let name = (path.Name() ?? "").lowercased()
        if name.hasSuffix(".gguf") {
            return try openGGUF(path)
        }
        if name.hasSuffix(".safetensors") {
            return try openSafetensors(path.Parent() ?? fs.Path("."), [path], path)
        }
        if isTorch(name) {
            return try openTorch(path.Parent() ?? fs.Path("."), [path], path)
        }
        throw OpenError.unsupported("\(path.String()) is not a .gguf, .safetensors or PyTorch file")
    }
    var shards: [fs.Path] = []
    var ggufs: [fs.Path] = []
    var pts: [fs.Path] = []
    for e in try fs.ReadDir(path) {
        let n = e.Name.lowercased()
        if n.hasSuffix(".safetensors") {
            shards.append(path / e.Name)
        } else if n.hasSuffix(".gguf") {
            ggufs.append(path / e.Name)
        } else if isTorch(n) && e.Kind != .directory {
            pts.append(path / e.Name)
        }
    }
    if let index = readJSON(path / "model.safetensors.index.json"), let map = index["weight_map"]?.Object {
        // The index names the files that hold the model; any others (a
        // consolidated copy) are not it.
        var files: [string] = []
        for (_, v) in map.Members {
            if let f = v.String, !files.contains(f) {
                files.append(f)
            }
        }
        shards = files.sorted().map { path / $0 }
    }
    if !shards.isEmpty {
        return try openSafetensors(path, shards.sorted { $0.String() < $1.String() }, path)
    }
    if ggufs.count == 1 {
        return try openGGUF(ggufs[0])
    }
    if ggufs.count > 1 {
        throw OpenError.unsupported("\(path.String()) holds \(ggufs.count) GGUF files; open the one wanted")
    }
    if !pts.isEmpty {
        return try openTorch(path, pts.sorted { $0.String() < $1.String() }, path)
    }
    throw OpenError.unsupported("\(path.String()) holds no safetensors, GGUF or PyTorch weights")
}

// isTorch is whether a file name is a PyTorch checkpoint's.
func isTorch(_ name: string) -> bool {
    return name.hasSuffix(".pth") || name.hasSuffix(".pt") || (name.hasPrefix("pytorch_model") && name.hasSuffix(".bin"))
}

// openTorch opens PyTorch checkpoints beside an optional config.json: a
// model whose config names no architecture (Kokoro's) opens with none.
func openTorch(_ dir: fs.Path, _ files: [fs.Path], _ path: fs.Path) throws -> Checkpoint {
    var cfg = readJSON(dir / "config.json")
    if cfg == nil {
        cfg = try json.Parse(bytes: Array("{}".utf8))
    }
    var arch = ""
    if let a = cfg!["architectures"]?[0]?.String {
        arch = a
    } else if let t = cfg!["model_type"]?.String {
        arch = t
    }
    var opened: [torch.File] = []
    var info: [string: TensorInfo] = [:]
    var names: [string] = []
    for i in 0..<files.count {
        let f = try torch.Open(files[i])
        for t in f.Tensors {
            if info[t.Name] != nil {
                throw OpenError.unsupported("\(t.Name) is in two files")
            }
            info[t.Name] = TensorInfo(Name: t.Name, Shape: t.Shape, DType: tensorType(t.DType),
                                      SourceType: t.DType.Name, Size: t.Size, file: i, offset: t.Offset)
            names.append(t.Name)
        }
        opened.append(f)
    }
    return Checkpoint(format: .torch, path: path, dir: dir, architecture: arch, config: Config(hf: cfg!),
                      st: [], pt: opened, gguf: nil, info: info, names: names)
}

func openGGUF(_ path: fs.Path) throws -> Checkpoint {
    let f = try gguf.Open(path)
    guard let arch = f.Architecture else {
        throw OpenError.unsupported("\(path.String()) has no general.architecture")
    }
    var info: [string: TensorInfo] = [:]
    var names: [string] = []
    for t in f.Tensors {
        // GGUF lists dimensions innermost first; a tensor, outermost.
        info[t.Name] = TensorInfo(Name: t.Name, Shape: t.Shape.reversed(), DType: tensorType(t.Type),
                                  SourceType: t.Type.Name, Size: t.Size, file: 0, offset: t.Offset)
        names.append(t.Name)
    }
    return Checkpoint(format: .gguf, path: path, dir: path.Parent(), architecture: arch, config: Config(gguf: f),
                      st: [], gguf: f, info: info, names: names)
}

func openSafetensors(_ dir: fs.Path, _ shards: [fs.Path], _ path: fs.Path) throws -> Checkpoint {
    guard let cfg = readJSON(dir / "config.json") else {
        throw OpenError.unsupported("no config.json beside the safetensors in \(dir.String())")
    }
    // A config may name no architecture (Kokoro's names none); a family
    // then claims the checkpoint by what else it holds.
    var arch = ""
    if let a = cfg["architectures"]?[0]?.String {
        arch = a
    } else if let t = cfg["model_type"]?.String {
        arch = t
    }
    var files: [safetensors.File] = []
    var info: [string: TensorInfo] = [:]
    var names: [string] = []
    for i in 0..<shards.count {
        let f = try safetensors.Open(shards[i])
        for t in f.Tensors {
            if info[t.Name] != nil {
                throw OpenError.unsupported("\(t.Name) is in two shards")
            }
            info[t.Name] = TensorInfo(Name: t.Name, Shape: t.Shape, DType: tensorType(t.DType),
                                      SourceType: t.DType.Name, Size: t.Size, file: i, offset: t.Offset)
            names.append(t.Name)
        }
        files.append(f)
    }
    return Checkpoint(format: .safetensors, path: path, dir: dir, architecture: arch, config: Config(hf: cfg),
                      st: files, gguf: nil, info: info, names: names)
}

func tensorType(_ t: gguf.TensorType) -> tensor.DType? {
    switch t {
    case .F32: return .F32
    case .F16: return .F16
    case .BF16: return .BF16
    case .Q4_0: return .Q4_0
    case .Q8_0: return .Q8_0
    case .Q4_K: return .Q4_K
    case .Q6_K: return .Q6_K
    default: return nil
    }
}

func tensorType(_ t: safetensors.DType) -> tensor.DType? {
    switch t {
    case .F32: return .F32
    case .F16: return .F16
    case .BF16: return .BF16
    default: return nil
    }
}

func tensorType(_ t: torch.DType) -> tensor.DType? {
    switch t {
    case .F32: return .F32
    case .F16: return .F16
    case .BF16: return .BF16
    default: return nil
    }
}
