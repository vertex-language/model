# model/safetensors

Reads safetensors, the Hugging Face Hub's canonical weight format: an 8-byte little-endian header length, a JSON header naming each tensor's dtype, shape and byte range, then the tensors' raw little-endian bytes.

```vertex
import "model/safetensors"
```

## Types

- **`DType`** (enum): DType is a tensor's element type, as the format names it.
- **`TensorInfo`** (struct): TensorInfo is where a tensor is and what it holds.
- **`FormatError`** (enum): FormatError is a file this refuses, and why.
- **`File`** (class): File is an open safetensors file: its tensors, its metadata, and the mapping their bytes are read from, held as long as the File is.
- **`Entry`** (struct): Entry is a tensor to write: its name, dtype, shape and bytes.

## Functions

- `func Open(_ path: fs.Path) throws -> File`: Open maps the safetensors file at path and reads its header. Tensor bytes are not read until they are used.
- `func Encode(_ entries: [Entry], metadata: [string: string] = [:]) throws -> [uint8]`: Encode is a safetensors file of entries, in their order, and metadata.
- `func Save(_ entries: [Entry], to path: fs.Path, metadata: [string: string] = [:]) throws`: Save writes entries to path as a safetensors file.

Part of the [`model`](https://github.com/vertex-language/model) repository.
