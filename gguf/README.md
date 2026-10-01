# model/gguf

Reads GGUF, the self-contained weight format of llama.cpp: metadata (architecture, hyperparameters, the tokenizer, the chat template) and tensors, quantized or not, in one file.

```vertex
import "model/gguf"
```

## Types

- **`TensorInfo`** (struct): TensorInfo is where a tensor is and what it holds.
- **`File`** (class): File is an open GGUF file: its metadata, its tensors, and the mapping their bytes are read from, held as long as the File is.
- **`TensorType`** (enum): TensorType is how a tensor's elements are stored: a plain scalar, or a quantized block of BlockSize elements in TypeSize bytes.
- **`ValueType`** (enum): ValueType is the type of a metadata value, by GGUF's numbers.
- **`Value`** (enum): Value is one metadata value: a scalar, a string, or an array of either. Arrays do not nest, as llama.cpp's reader requires.
- **`FormatError`** (enum): FormatError is a file that is not well-formed GGUF, and says where.
- **`Tensor`** (struct): Tensor is a tensor to write: its name, type, shape in ggml's order (innermost first) and its bytes, laid out as the type lays them.
- **`Layout`** (struct): Layout is a tensor as the header describes it: its bytes come later.

## Functions

- `func Open(_ path: fs.Path) throws -> File`: Open maps the GGUF file at path and reads its header. Tensor bytes are not read until they are used.
- `func Header(metadata: [(string, Value)], tensors: [Layout]) throws -> [uint8]`: Header is a GGUF v3 file's header -- metadata (in order), then each tensor's name, shape, type and offset -- padded to where the data begins.
- `func Encode(metadata: [(string, Value)], tensors: [Tensor]) throws -> [uint8]`: Encode is a GGUF v3 file of metadata and tensors, in memory.
- `func Save(metadata: [(string, Value)], tensors: [Tensor], to path: fs.Path) throws`: Save writes a GGUF file.
- `func Save(metadata: [(string, Value)], tensors: [Layout], to path: fs.Path, bytes: (int) throws -> [uint8]) throws`: Save writes a GGUF file a tensor at a time: bytes(i) is tensor i's data, asked for as it is written, so a model larger than memory converts.

Part of the [`model`](https://github.com/vertex-language/model) repository.
