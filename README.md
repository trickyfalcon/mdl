# mdl — one local LLM library, every Windows runtime

`mdl` downloads a model **once**, in the formats you choose, and wires it up to every local
runtime with minimal duplication:

| Runtime | Reads | How mdl wires it |
| --- | --- | --- |
| **transformers / vLLM** | raw safetensors by repo id, from the HF cache | downloads into `HF_HOME` |
| **llama.cpp** (`llama-server.exe -m`) | a GGUF file by path | the GGUF master on the fast disk |
| **LM Studio** | GGUFs under `<publisher>\<model>\file.gguf` | points LM Studio's models folder at the GGUF master (no copy, no symlink) |
| **Ollama** | its own blob store | imports the GGUF via a Modelfile (`ollama create`) |

It is built for **native Windows + PowerShell** (paths, drives, `%VARS%`, `.exe` tools, no
symlinks), packaged and run with [`uv`](https://docs.astral.sh/uv/).

---

## The storage model (hot/cold, one copy each)

```
H:  (archive disk)   HF cache         raw safetensors        transformers / vLLM
D:  (fast disk)      <gguf_dir>        GGUF master (1 file)   llama.cpp + LM Studio (shared)
D:  (fast disk)      <ollama_models>   Ollama blob store      Ollama (unavoidable copy)
```

* **Raw safetensors** live once in the Hugging Face cache at `HF_HOME` on **H:** (the big,
  full-precision files on the archive disk). transformers/vLLM load them by repo id.
* **GGUF** is a single file shared by llama.cpp *and* LM Studio. The master lives on **D:**
  (the fast disk), in exactly the layout LM Studio requires:
  `<gguf_dir>\<publisher>\<model>\<file>.gguf`.
  * llama.cpp loads it directly: `llama-server.exe -m <path>`.
  * **LM Studio** — *no symlink, no copy*. Because mdl already places GGUFs in
    `<publisher>\<model>\file.gguf` form, you simply point LM Studio's models folder at
    `<gguf_dir>` and it lists them with zero duplication. `mdl doctor` checks whether LM
    Studio's folder equals `gguf_dir` and tells you how to fix it if not.
  * **Ollama** *cannot* read a loose GGUF. It imports via a Modelfile (`FROM <abs path>`) and
    `ollama create` **copies** the GGUF into its blob store. That copy is unavoidable; to keep
    it on the fast disk, set `OLLAMA_MODELS` to a folder on **D:** (doctor checks this).

> **Why no symlinks?** Creating file symlinks on Windows needs admin or Developer Mode, and the
> GGUF master on D: can't be hard-linked into C: (cross-volume). So mdl never links — it places
> files where each runtime already looks.

---

## Install (Windows + uv)

```powershell
# 1. install uv (user scope, no admin) if you don't have it
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. from the project folder, create the venv + install (uv fetches Python 3.12 itself)
uv sync

# 3. run it
uv run mdl --help
```

`uv sync` builds a console entry point, so within the project `uv run mdl ...` works. To get a
bare `mdl` on your PATH, install it as a uv tool:

```powershell
uv tool install --from . mdl     # then `mdl` works from any shell
```

> **Note:** keep this project *outside* `Documents`/`Desktop`/`Pictures` if you have Windows
> **Controlled Folder Access** (Defender ransomware protection) enabled — it silently blocks
> writes there. mdl's own stores (H:, D:, `%APPDATA%\mdl`) are unaffected.

---

## Configuration

Config lives at **`%APPDATA%\mdl\config.toml`** (resolved via `platformdirs`). It's created with
sensible defaults on first run. Paths are written as **TOML literal strings (single quotes)** so
Windows backslashes are stored verbatim, and `%VARS%` / a leading `~` are expanded at runtime.

```toml
hf_home        = 'H:\models\hf'                                          # raw safetensors (transformers/vLLM)
gguf_dir       = 'D:\models\gguf'                                        # GGUF master (llama.cpp + LM Studio)
lmstudio_dir   = '%USERPROFILE%\.lmstudio\models'                        # doctor's fallback if LM Studio's settings.json can't be read
ollama_models  = 'D:\models\ollama'                                      # advise OLLAMA_MODELS = this
ollama_bin     = 'ollama'
llamacpp_dir   = 'C:\src\llama.cpp'
llama_quantize = 'C:\src\llama.cpp\build\bin\Release\llama-quantize.exe' # adjust to your build layout
default_quant  = 'Q4_K_M'
```

```powershell
mdl config                       # show config (raw + expanded)
mdl config set gguf_dir 'E:\gguf'
```

### Recommended environment variables

```powershell
setx HF_HOME "H:\models\hf"        # so transformers/vLLM share mdl's cache (new shells only)
setx OLLAMA_MODELS "D:\models\ollama"   # keep Ollama's blob copies on the fast disk; restart Ollama after
```

(mdl injects `HF_HOME` and `HF_XET_HIGH_PERFORMANCE=1` into its own download subprocesses, so you
don't strictly need them set globally — but setting `HF_HOME` makes your other tools agree.)

---

## Commands

```
mdl add <hf_repo>     download once + wire up runtimes (the headline command)
mdl list              table: formats, quants, runtimes, drive + size per format, totals
mdl rm <model>        remove across stores / deregister runtimes
mdl sync              re-apply all registrations from config (after moving the library)
mdl convert <src>     standalone safetensors -> GGUF
mdl config [set ...]  show / edit config
mdl doctor            Windows-aware status of drives, tools, env vars, runtimes
```

Global flags: `--dry-run` (print the plan, change nothing), `--verbose/-v` (stream subprocess
output). Both go *before* the subcommand: `mdl --dry-run add ...`.

### `mdl add`

```powershell
# prebuilt GGUF from a community repo + full safetensors, register everywhere
mdl add Qwen/Qwen3-32B --gguf-repo bartowski/Qwen3-32B-GGUF --quant Q4_K_M

# no prebuilt GGUF exists -> build it locally from the safetensors
mdl add some-org/brand-new-model --convert --quant Q5_K_M

# GGUF only (skip the big raw download), Ollama only
mdl add Qwen/Qwen3-32B --gguf-repo bartowski/Qwen3-32B-GGUF --no-raw --register ollama
```

| Option | Meaning |
| --- | --- |
| `--gguf-repo TEXT` | repo holding the GGUFs (e.g. `bartowski/<model>-GGUF`). If omitted, mdl searches the Hub; if none found and `--convert` is set, it builds one. |
| `--quant TEXT` | quant to pull/build (default from config). Pulls only `*<quant>*` via `hf download --include`. |
| `--raw/--no-raw` | also download full safetensors into `HF_HOME` (default on). |
| `--gguf/--no-gguf` | download/place the GGUF on D: (default on). |
| `--convert` | build the GGUF locally if no prebuilt repo. Add `--remote <repo>` to stream weights from the Hub instead of downloading first. |
| `--register TEXT` | runtimes to wire up, csv (default `ollama,lmstudio`; use `none` to skip). |

`add` is **idempotent**: present downloads are skipped, existing Ollama models are not
re-created, structure is re-verified safely.

### `mdl rm`

```powershell
mdl rm Qwen/Qwen3-32B                       # everything: raw + gguf + ollama + record
mdl rm Qwen/Qwen3-32B --from ollama         # just deregister from Ollama (keep files)
mdl rm Qwen/Qwen3-32B --format gguf         # delete only the GGUF master
mdl rm Qwen/Qwen3-32B --yes                 # skip the confirmation
```

It prints exactly what will be deleted, on which drive, and how much space is freed, then asks
for confirmation (unless `--yes` or `--dry-run`).

### `mdl sync`

Re-applies every recorded registration from the **current** config. This is what you run after
moving the library — e.g. you copied `D:\models` to a NAS or changed drive letters:

```powershell
mdl config set gguf_dir '\\nas\models\gguf'
mdl config set ollama_models 'N:\ollama'
mdl sync         # re-imports into Ollama, re-verifies LM Studio structure
```

### `mdl convert`

```powershell
mdl convert Qwen/Qwen3-32B --quant Q4_K_M           # download raw, then convert + quantize
mdl convert Qwen/Qwen3-32B --remote --quant Q4_K_M  # stream weights from the Hub (no full download)
mdl convert 'D:\some\local\model_dir' --quant Q5_K_M
```

Conversion runs `python <llamacpp_dir>\convert_hf_to_gguf.py` to make an f16 GGUF, then
`llama-quantize.exe` to the requested quant. The converter's heavy deps (torch/transformers) are
supplied on demand via `uv run --with-requirements <llamacpp>\requirements\...`, so they never
pollute mdl. Needs llama.cpp present (see below); `doctor` tells you if it isn't.

---

## `mdl doctor`

Checks, Windows-aware, with concrete fixes:

* drives for `hf_home` and `gguf_dir` exist and are writable;
* `huggingface_hub` importable and the `hf` CLI present; whether you're logged in (gated models);
* `ollama.exe` found and the service responding;
* `OLLAMA_MODELS` set and on D: (warns if unset or on C:);
* `llamacpp_dir` has `convert_hf_to_gguf.py` and the configured `llama-quantize.exe`;
* LM Studio's models folder (read from `~/.lmstudio/settings.json`) equals `gguf_dir`;
* `HF_HOME` env var matches config (so transformers/vLLM share the cache).

---

## Building llama.cpp (for `--convert`)

`mdl` needs `llama-quantize.exe` only for conversion. The included
`scripts\build_llamacpp.ps1` builds it with **no admin**: CMake + Ninja via `uv tool install`,
and the portable MinGW GCC toolchain (w64devkit). It produces
`<llamacpp_dir>\build\bin\llama-quantize.exe` (Ninja single-config layout — note: an MSVC build
puts it under `build\bin\Release\`). Point `llama_quantize` in config at whatever you build.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_llamacpp.ps1
mdl config set llama_quantize 'C:\src\llama.cpp\build\bin\llama-quantize.exe'
```

---

## vLLM / WSL

vLLM is Linux-oriented. If you later run it under **WSL**, it can read the same raw cache on H:
as `/mnt/h/models/hf` (set `HF_HOME=/mnt/h/models/hf` inside WSL) — no second download. The
native runtimes here (Ollama, LM Studio, llama.cpp) all have Windows builds and need no WSL.

---

## Development

```powershell
uv run pytest        # unit tests for the pure logic
```

Tested pure logic includes: `%VAR%`/`~` path expansion, the LM Studio target path from a repo
id, Modelfile content/quoting, Ollama name derivation, literal-TOML round-tripping, and the
convert/quantize command builders.

### Layout

```
src/mdl/
  cli.py            Typer commands + friendly error handling
  config.py         defaults, literal-TOML load/save, platformdirs
  paths.py          path expansion + repo/target/size helpers (pure, tested)
  hub.py            hf download subprocess + Hub metadata/discovery
  convert.py        convert_hf_to_gguf.py + llama-quantize.exe (pure command builders)
  add.py            the add orchestration
  ops.py            rm + sync
  library.py        %APPDATA%\mdl\library.json manifest + live inventory
  doctor.py         status checks
  registry/ollama.py    Modelfile build + ollama create/rm
  registry/lmstudio.py  structure verify + models-folder advice (no symlinks)
```
