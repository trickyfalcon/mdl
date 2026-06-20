# mdl — handoff notes

Context for picking the work back up (e.g. a fresh Claude Code session on another machine).
`mdl` manages one local LLM library across HF transformers/vLLM, llama.cpp, Ollama, and
LM Studio. It downloads a model once and wires it into every runtime.

## Current state (as of this handoff)

All of the following is **merged to `main`** (PR "Resilient downloads + cross-platform
support"). **89 tests passing** via `uv run pytest`.

Shipped:
- **Live download progress** — streams `hf download` so its native bar renders (it hides the
  bar whenever stdout is captured).
- **Resume + dedup** — `mdl add` checks on-disk bytes/files vs the Hub first: skips when
  complete, resumes when partial; `--force` overrides.
- **`--retries N`** + **stall watchdog** — retries transient failures with backoff and kills
  + resumes a download whose on-disk bytes flatline for `download_stall_timeout` seconds
  (covers silent xet hangs). `HF_HUB_DOWNLOAD_TIMEOUT` set for the classic path.
- **Disk-space pre-flight** warning before large pulls.
- **`mdl list --check`** — Hub-verified completeness (OK / NN%).
- **`mdl verify <model> [--repair] [--retries N]`** — check vs Hub, re-pull gaps; non-zero
  exit when incomplete.
- **`mdl gc [model] [--locks] [--force] [-y]`** — reclaim abandoned `*.incomplete` partials.
- **Concurrent-run lock** — two `mdl add` for the same repo can't race.
- **Convert/quantize** output now streams too.
- **Cross-platform** — `osenv.py` centralizes per-OS defaults (Windows drives vs
  `~/models/...`), `.exe` handling, and `setx` vs `export` hints. Windows unchanged.

New config keys: `download_timeout` (30s), `download_stall_timeout` (300s).

## Where things live (fast ramp for a new session)

- `src/mdl/osenv.py` — all OS differences (defaults, exe names, env hints). Start here for
  anything platform-specific.
- `src/mdl/proc.py` — subprocess runner: `stream=True` (inherit terminal → live bar) and the
  stall watchdog (`_run_streamed` / `_kill_tree`).
- `src/mdl/hub.py` — `hf` downloads, `_run_download` retry loop, `raw_status`/`gguf_status`
  (Hub reconciliation), `classify_download`.
- `src/mdl/add.py` — `add_model` (lock wrapper) → `_add_model_locked` (the flow).
- `src/mdl/verify.py`, `src/mdl/ops.py` (rm/sync/**gc**), `src/mdl/locks.py`.
- Config: `%APPDATA%\mdl\config.toml` (Win) / `~/.config/mdl/config.toml` (POSIX), via
  platformdirs.

## macOS validation — DONE (2026-06-20, Apple Silicon, macOS 15 / Darwin 25.5)

Ran the full ramp on real Mac hardware. Everything POSIX works:
- `uv sync` clean; `uv run mdl doctor` — drives resolve to `~/models/hf` + `~/models/gguf`
  (both OK/writable), `hf` CLI found. Expected WARN/FAIL on a bare box: no ollama, no
  llama.cpp, not HF-logged-in, `HF_HOME`/`OLLAMA_MODELS` unset, LM Studio dir mismatch.
- `mdl add HuggingFaceTB/SmolLM2-135M-Instruct --no-gguf --register none` — downloaded to
  the HF cache; re-run correctly **skipped** (dedup). `list` / `list --check` (Hub-verified
  **OK**) / `verify` (**100% complete**) / `rm` all behaved. Smoke model removed afterward.
- `uv run pytest` — **88 passed, 5 skipped** on macOS.

**Test fix shipped this session:** `tests/test_paths.py` had 5 failures on POSIX — they
asserted Windows-only path semantics (`%VAR%` expansion, drive letters, case-insensitive
backslash paths) that the OS-delegating helpers can't reproduce off Windows. Gated those with
`@windows_only` and added `@posix_only` companions (`$VAR`/`~` expansion, empty drive, POSIX
`same_path`). The production code was already correct — this was a test-portability gap.
Windows still runs its original 89 (now 4 of them skip there).

Watch-items confirmed fine on macOS:
- LM Studio dir default `~/.lmstudio/models` is correct (LM Studio not installed here, so
  untested live, but the path/default is right).
- `uv` / `hf` discoverable under zsh PATH; `ollama` not installed (expected FAIL).
- Stall watchdog tree-kill on POSIX is `terminate → kill` (fine for single-process `hf`).

## After that: NAS phase (own branch, e.g. `feature/nas`)

Goal: one library on a self-hosted NAS, shared across Windows/Mac/Linux. **No cloud** — a
deliberate decision (re-hosting public HF models on S3/Azure is redundant cost + a provider
you don't control; HF is already the free source of truth). NAS = a mounted path, so the work
is path-robustness, not a storage SDK:
- UNC / network paths (`\\synology\share`, mapped drives, `/mnt/nas`) in `drive_letter` /
  `free_space` / doctor.
- Mount-health check in `doctor` so `add` won't write 800 GB into a dead mount point.
- Testable against the existing Synology SMB share before the planned Unraid build.

## Loose end (Windows-only, not relevant to Mac)

A `deepseek-ai/DeepSeek-V4-Pro` raw download is paused on the Windows box: ~727 GB present on
`H:`, 5 `*.incomplete` partials (~39 GB), ~78 GB remaining. Resume with the new code:
`hf auth login` then `uv run mdl add deepseek-ai/DeepSeek-V4-Pro --no-gguf --register none --retries 10`.
Do **not** `mdl gc` it unless abandoning — those partials are its resume data.
