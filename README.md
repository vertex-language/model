# model

[![package: vs-package](https://img.shields.io/badge/package-vs--package-f4f4f5?style=flat-square&labelColor=e4e4e7&color=18181b)](https://github.com/vertex-language)
[![formats: safetensors | gguf | torch](https://img.shields.io/badge/formats-safetensors%20%7C%20gguf%20%7C%20torch-f4f4f5?style=flat-square&labelColor=e4e4e7&color=18181b)](https://github.com/vertex-language/model)

What every model shares, and no model: weight formats, and a checkpoint that reads any of them one way. The models themselves live in their task's repository: [`llm`](https://github.com/vertex-language/llm) for text generation, [`tts`](https://github.com/vertex-language/tts) for speech synthesis. The layout is `proposed_vs_model.md`.

---

## Quick Start

```bash
vsc run test-gguf           # the GGUF reader and writer against llama.cpp's
vsc run test-safetensors    # the safetensors reader and writer against the reference
vsc run test-quant          # every quantized tensor decoded, bit for bit
vsc run test-torch          # the PyTorch reader against torch.load, Kokoro-82M included
```

---

## Packages

| Package | What it is |
| --- | --- |
| **`model`** | `model.Open(path)`: a Hugging Face snapshot directory (config.json and safetensors, sharded or not, or PyTorch weights) or a GGUF file, opened offline as a `Checkpoint`. A config naming no architecture (Kokoro's) opens with none, for a family to claim by what else it holds. It answers in Hugging Face's vocabulary whatever the format: `ck.Config.Int("hidden_size")`, `ck.Tensor("model.layers.0.self_attn.q_proj.weight", on: device)` (a zero-copy view of the mapping where the device reads it in place), `ck.Tokenizer()`, `ck.ChatTemplate`, `ck.Architecture`. How a task's names are spelled in a GGUF is the task's to give (`ck.Aliases`; `llm/arch.DecoderAliases` for decoders) |
| **`model/fetch`** | `fetch.Open(ref)`: a Hub reference (`hf.co/org/name`, `…:Q4_K_M`, a file in a repo) or a path. The repository's files are listed first (no bytes), a format is chosen, and only its files are downloaded through `remote/hub`. `fetch.Inspect(ref)` says what a reference is without loading it. The only package here that reaches the network: families never import it |
| **`model/gguf`** | `gguf.Open(path)`: GGUF v2/v3, memory-mapped; typed metadata, tensor infos, zero-copy bytes. `gguf.Save` and `gguf.Encode` write one |
| **`model/safetensors`** | `safetensors.Open(path)`: the header parsed, the data mapped, nothing copied. `safetensors.Save` and `Encode` write one |
| **`model/torch`** | `torch.Open(path)`: a PyTorch checkpoint (`.pth`, `.pt`, `pytorch_model.bin`; the zip format), its pickle read by a restricted unpickler that refuses any global but tensor rebuilding, as `torch.load(weights_only=True)` does; tensors mapped in place, contiguous ones only. `model.Open` opens a directory of them as a checkpoint |

Quantized GGUF tensors are decoded on a device by `gpu/dtype` (`dtype.Dequantize`).

---

## Testing

Each format is held to its own reference. `testdata/oracle/gguf_dump.cpp`
prints what ggml's `gguf.h` reads from a file, with floats as their bits and
arrays and tensor bytes as FNV-1a hashes (`cmd/test-gguf/golden`);
`testdata/oracle/safetensors_dump.py` does the same with the `safetensors`
library (`cmd/test-safetensors/golden`). To regenerate the GGUF dumps from a
llama.cpp checkout built with CMake:

```console
$ c++ -std=c++17 -I$LLAMA/ggml/include testdata/oracle/gguf_dump.cpp -L$LLAMA/build/bin -lggml-base -Wl,-rpath,$LLAMA/build/bin -o gguf_dump
$ ./gguf_dump testdata/stories260K.gguf > cmd/test-gguf/golden/stories260K.txt
```

The test files live in `testdata/` and are not committed: llama2.c's
"tinyllamas" as GGUF, from `hf.co/ggml-org/models-moved` under `tinyllamas/`,
and `hf.co/hf-internal-testing/tiny-random-LlamaForCausalLM` as
`testdata/tiny-llama`:

```console
$ for f in stories260K.gguf stories15M-q4_0.gguf stories110M-q4_k_m.gguf; do curl -sSL -o testdata/$f https://huggingface.co/ggml-org/models-moved/resolve/main/tinyllamas/$f; done
```

---

## License

[MIT](LICENSE)
