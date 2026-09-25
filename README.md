# model

Running models in Vertex: model families, the kits they share, weight
formats, and generation. The design is `proposed_ai_packages.md` §6.3 and §10.
The first target is `model/llama`, running a quantized GGUF llama end to end.
Every package it needs is built as the path reaches it.

| Package | Built | Tested |
| --- | --- | --- |
| `model/gguf` | `gguf.Open(path)`: a GGUF v2/v3 file, memory-mapped through `fs/mmap`. The header is read, and tensor bytes are used where they lie (`File.Bytes`) or copied (`File.Copy`). Typed metadata accessors (`Text`, `Integer`, `Number`, `Flag`, `Texts`, `Integers`, `Numbers`), `TensorInfo` (name, shape innermost first, `TensorType`, offset, size), and every ggml tensor type with its block and byte size | both test models dumped line for line as llama.cpp's own reader dumps them, tensor bytes hashed; 20 malformed files, each refused for llama.cpp's reason |
| `model/llama` | The Llama family (`general.architecture` `llama`): `Config.Read` from a GGUF's `llama.*` keys, `Model.Load(path, on: device)` (weights by their GGUF names, a tied head when there is no `output.weight`, the tokenizer from `tokenizer.ggml.*`), `Forward(token, position)` to logits with a KV cache, and greedy `Generate`. f32, Q4_0 and Q8_0 weights; grouped-query attention | `test-llama`: "Once upon a time" through both test models on the CPU device and Metal. The 32 greedy tokens of each match llama.cpp's exactly. Logits are within 0.01 (f32) and 0.25 (q4_0) of llama.cpp's, and the f32 model's first step is within 1e-4 of float64 math (2.1e-6 in fact; llama.cpp's is 2.5e-4 off) |
| (with `gpu/dtype`) | Every quantized tensor of a GGUF file decoded on a device: `dtype.Dequantize(bytes, dtype.Q4_0(), …)` | `test-quant`: the 44 Q4_0 and Q8_0 tensors of `stories15M-q4_0`, on the CPU device and Metal, bit for bit what llama.cpp's `to_float` makes (`tests/quant/golden`, from `gguf_dump -dequant`) |

## Speed

`vsc run bench` loads `stories15M-q4_0` and generates 100 tokens on each
device. `vsc run profile` splits a token's time into encoding its launches,
the device running them, and reading the logits. `vsc run ops` times each
operation alone at the model's sizes. On an Apple M-series Mac
(2026-09-25):

| | Vertex | llama.cpp (same machine) |
| --- | --- | --- |
| Metal | ~320 tokens/s | 615 tokens/s |
| CPU | ~70 tokens/s | 3,433 tokens/s |

A model this small is bound by launches, not bandwidth: ~60 kernels a
token. Fusing more of them and repacking weights for the device are next.

## Testing against llama.cpp

llama.cpp is the oracle, cloned and built from source. `tests/oracle/llama_run.cpp` runs a prompt through libllama a token at a time (CPU, float32 KV cache) and prints each step's top logits and the greedy continuation (`tests/llama/golden`). `tests/oracle/gguf_dump.cpp` prints what ggml's
`gguf.h` reads from a file, with floats as their bits and arrays and tensor
bytes as FNV-1a hashes. Its output is in `tests/gguf/golden`. To regenerate
it from a llama.cpp checkout built with CMake:

```console
$ c++ -std=c++17 -I$LLAMA/ggml/include tests/oracle/gguf_dump.cpp -L$LLAMA/build/bin -lggml-base -Wl,-rpath,$LLAMA/build/bin -o gguf_dump
$ ./gguf_dump testdata/stories260K.gguf > tests/gguf/golden/stories260K.txt
```

## Test models

These live in `testdata/` and are not committed. They are llama2.c's
"tinyllamas" converted to GGUF, from `hf.co/ggml-org/models-moved`, under
`tinyllamas/`:

| File | Size | What it tests |
| --- | --- | --- |
| `stories260K.gguf` | 1.2 MB | f32 weights; grouped-query attention (8 heads, 4 KV heads); a 512-token vocabulary |
| `stories15M-q4_0.gguf` | 19 MB | Q4_0 weights and a Q8_0 output head; a 32000-token SentencePiece vocabulary |

```console
$ for f in stories260K.gguf stories15M-q4_0.gguf; do curl -sSL -o testdata/$f https://huggingface.co/ggml-org/models-moved/resolve/main/tinyllamas/$f; done
$ vsc run test-gguf
$ vsc run test-quant
$ vsc run test-llama
```
