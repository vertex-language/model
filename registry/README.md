# model/registry

Models by name -- `qwen3`, `qwen3:8b`, `qwen3:8b-q8_0` -- each pinned to one repository on the Hugging Face Hub at one commit, with the size and hash of every file it is made of.

386 models: 331 `llm`, 25 `diffusion`, 18 `embed`, 11 `asr`, 1 `tts`. Being on the list is a promise that the files download without a login, not that a family runs them: the list is the ladder of known models that support is measured against, one model at a time.

```vertex
import "model/registry"

let model = try await registry.Pull("qwen3:8b")
print(model.Path)          // …/snapshots/<commit>/Qwen3-8B-Q4_K_M.gguf
```

```bash
vsc run registry -- list llm             # every model, or one task's
vsc run registry -- show qwen3:8b        # repository, commit, quants, files
vsc run registry -- pull qwen3:8b-q8_0   # into the Hugging Face cache
vsc run registry -- probe qwen3:8b       # the first bytes of each file, anonymously
vsc run test-registry                    # the list checked, offline
```

## Names

| Written | Means |
| --- | --- |
| `qwen3` | `qwen3:latest` |
| `qwen3:latest` | the name's most used tag, at its default quant |
| `qwen3:8b` | that tag, at its default quant (`q4_k_m` where there is one) |
| `qwen3:8b-q8_0` | that tag and quant |
| `qwen2.5:3b-base` | a base model, beside its instruct twin `qwen2.5:3b` |

The generation is part of the name (`qwen2.5`, `qwen3`, `gemma3`, `gemma4`), so `latest` never moves to a new architecture.

## What is on the list

- **Originals only.** No fine-tunes, merges, distills or re-uploads of another lab's model, by its card's `base_model` chain and by its name.
- **Once each.** One entry per original, from the first that has it without a login: unsloth's GGUF repositories, the author's own, ggml-org, a plain quant mirror.
- **Every generation and size**, oldest to newest.
- **From-scratch models with no GGUF** (Kokoro) carry their own files, as one quant named `files`.

To add a model, add an `Entry` to `table.vs`; its comment says where the pins come from.

## Errors

| Error | Means |
| --- | --- |
| `unknownName` | Not on the list; the message suggests names that look like it |
| `unknownTag` | A known name without that tag or quant; the message lists its tags |
| `needsLogin` | The Hub wants a login for it now (gated, private, 401/403). Set `HF_TOKEN` to a read token, or run `hf auth login` |
| `rateLimited` | Without a login the Hub allows an address 500 API calls in five minutes. Wait, or set `HF_TOKEN` |
| `changed` | The repository no longer holds what was pinned; the entry needs a new pin |

`Pull` makes one API call, then downloads; `remote/hub` checks every file against its hash.

## Types

- **`Entry`** (struct): one model: name, tag, task, the repository and commit it comes from, its source, architecture, license, quants and projector.
- **`Quant`** (struct): one quantization (`q4_k_m`) and its files: one, or the parts of a split GGUF.
- **`File`** (struct): path, size, and sha256 (or git blob id for a small file).
- **`Pick`** (struct): what a reference names: an entry and one of its quants.
- **`Pulled`** (struct): a pick whose files are in the cache, and the path to open.
- **`Probed`** (struct): what asking for a file's first bytes found.
- **`RegistryError`** (enum): the errors above.

## Functions

- `func Entries() -> [Entry]`: the whole list, by task, name and tag.
- `func Names() -> [string]`: the list's names, each once.
- `func Find(_ ref: string) throws -> Pick`: what a reference names.
- `func Suggest(_ name: string) -> [string]`: up to eight names that look like name.
- `func Ref(_ p: Pick) -> hub.Ref`: the hub reference a pick downloads from, at its pinned commit.
- `func Pull(_ ref: string, hub h: hub.Hub = hub.Hub()) async throws -> Pulled`: download what a reference names, only what is missing.
- `func Check(_ p: Pick, _ listed: [hub.RepoFile]) throws`: whether a repository's listed files hold what was pinned.
- `func IsPulled(_ p: Pick, cache: hub.Cache = hub.Cache.Default()) -> bool`: whether a pick's files are all in the cache.
- `func Probe(_ p: Pick, endpoint: string = "https://huggingface.co") async throws -> [Probed]`: the first four bytes of each file, asked for as a stranger would.

Part of the [`model`](https://github.com/vertex-language/model) repository.
