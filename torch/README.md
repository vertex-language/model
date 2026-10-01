# model/torch

Reads PyTorch checkpoints -- the zip files torch.save writes (.pth, .pt, pytorch_model.bin) -- as tensors by name, the file mapped and nothing copied.

```vertex
import "model/torch"
```

## Types

- **`DType`** (enum): DType is a storage's element type.
- **`TensorInfo`** (struct): TensorInfo is a tensor in the file: its name, shape (outermost first), element type, and where its bytes lie in the file.
- **`FormatError`** (enum): FormatError is a file that is not a checkpoint this reads, and why.
- **`File`** (class): File is an opened checkpoint.

## Functions

- `func Open(_ path: fs.Path) throws -> File`: Open maps a PyTorch checkpoint and reads its tensors' names, shapes and places; no tensor byte is read.

Part of the [`model`](https://github.com/vertex-language/model) repository.
