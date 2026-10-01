# model/fetch

Opens a model by reference: a local file or directory, or a Hugging Face Hub reference whose files are fetched into the Hugging Face cache first.

```vertex
import "model/fetch"
```

## Types

- **`FetchError`** (enum): FetchError is a reference that could not be opened, and why.
- **`Choice`** (struct): Choice is the format chosen for a Hub repository and the file patterns that fetch it.
- **`Info`** (struct): Info is what a reference is, read without loading it.

## Functions

- `func Open(_ ref: string, hub h: hub.Hub = hub.Hub()) async throws -> model.Checkpoint`: Open is the checkpoint a reference names, its files fetched when it is a Hub reference.
- `func IsLocal(_ ref: string) -> bool`: IsLocal is whether a reference is a path on this machine rather than a Hub reference: it starts with "/", "./", "../" or "~", or names something that exists.
- `func Choose(_ files: [hub.RepoFile], ref: hub.Ref) throws -> Choice`: Choose picks a Hub repository's format from its file list: GGUF for a quant tag or a repository of nothing else; else safetensors, fetching the top-level config, weights and tokenizer files only -- not a PyTorch copy, ONNX, or Meta's original/ -- which is all a checkpoint reads.
- `func Inspect(_ ref: string, keys: [string] = [], aliases: model.Aliases = model.Aliases()) async throws -> Info`: Inspect opens a reference -- fetching its files if it is on the Hub -- and says what it is: architecture, format, size and the config keys asked for, read through aliases (a task's spellings for a GGUF), with nothing put on a device.

Part of the [`model`](https://github.com/vertex-language/model) repository.
