"""Per-OS defaults and small platform helpers, so the rest of mdl stays OS-agnostic.

Everything that differs between Windows / macOS / Linux is funnelled through here: the starter
config (paths, exe names), how to name an executable, and how to suggest persisting an env var.
Only the path values differ across OSes; tunables (quant, timeouts) are shared.
"""

from __future__ import annotations

import os

IS_WINDOWS = os.name == "nt"


def exe(name: str) -> str:
    """Executable filename for this OS (``hf`` -> ``hf.exe`` on Windows, ``hf`` elsewhere)."""
    return f"{name}.exe" if IS_WINDOWS else name


# Windows leans on separate drives (fast NVMe vs bulk disk); POSIX mirrors the same split
# under the home dir. Either is just a starting point -- `mdl config set` repoints anything,
# e.g. at a NAS mount (`/mnt/nas/models`, `Z:\models`, `\\server\share\models`).
_WINDOWS_DEFAULTS: dict[str, str] = {
    "hf_home": r"H:\models\hf",
    "gguf_dir": r"D:\models\gguf",
    "lmstudio_dir": r"%USERPROFILE%\.lmstudio\models",
    "ollama_models": r"D:\models\ollama",
    "ollama_bin": "ollama",
    "llamacpp_dir": r"C:\src\llama.cpp",
    "llama_quantize": r"C:\src\llama.cpp\build\bin\Release\llama-quantize.exe",
    "default_quant": "Q4_K_M",
    "download_timeout": "30",
    "download_stall_timeout": "300",
}

_POSIX_DEFAULTS: dict[str, str] = {
    "hf_home": "~/models/hf",
    "gguf_dir": "~/models/gguf",
    "lmstudio_dir": "~/.lmstudio/models",
    "ollama_models": "~/.ollama/models",
    "ollama_bin": "ollama",
    "llamacpp_dir": "~/src/llama.cpp",
    "llama_quantize": "~/src/llama.cpp/build/bin/llama-quantize",
    "default_quant": "Q4_K_M",
    "download_timeout": "30",
    "download_stall_timeout": "300",
}


def default_config() -> dict[str, str]:
    """The starter config for the current OS (a fresh copy each call)."""
    return dict(_WINDOWS_DEFAULTS if IS_WINDOWS else _POSIX_DEFAULTS)


def env_set_hint(var: str, value: str) -> str:
    """A copy-pasteable command to persist an env var on this OS."""
    if IS_WINDOWS:
        return f'setx {var} "{value}"'
    return f'export {var}="{value}"   (add to your ~/.bashrc or ~/.zshrc)'
