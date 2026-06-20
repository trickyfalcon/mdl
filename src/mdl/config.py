"""Configuration: defaults, load/save, and the literal-TOML writer.

The config lives at ``%APPDATA%\\mdl\\config.toml`` (resolved via ``platformdirs`` rather than
hardcoded). Paths are stored as raw strings -- possibly containing ``%VARS%`` -- and expanded
on access via :func:`mdl.paths.expand_path`.

Critically, we *write* paths as TOML **literal strings** (single quotes) so a Windows path
like ``D:\\models\\gguf`` is stored verbatim instead of having its backslashes treated as
escape sequences. ``tomllib`` (stdlib) reads both literal and basic strings fine, so only the
writer is custom.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import platformdirs

from .errors import ConfigError
from .osenv import default_config
from .paths import expand_path

# --- defaults (per-OS: Windows uses drive letters, POSIX mirrors under ~; see osenv) ------
DEFAULTS: dict[str, str] = default_config()

#: keys whose values are filesystem paths (expanded + drive-checked)
PATH_KEYS = frozenset(
    {"hf_home", "gguf_dir", "lmstudio_dir", "ollama_models", "llamacpp_dir", "llama_quantize"}
)

CONFIG_DIR: Path = Path(platformdirs.user_config_dir("mdl", appauthor=False, roaming=True))
CONFIG_PATH: Path = CONFIG_DIR / "config.toml"


def _int_or(value: str, default: int) -> int:
    """Parse a config string as a non-negative int, falling back to ``default``."""
    try:
        n = int(str(value).strip())
        return n if n >= 0 else default
    except (TypeError, ValueError):
        return default


def _dump_value(value: str) -> str:
    """Serialize a string as a TOML literal string when safe, else a basic string.

    Literal strings (single quotes) can't contain a single quote, newline or carriage
    return; paths never do, so they always take the literal branch and keep their
    backslashes verbatim. The basic-string fallback exists only for pathological values.
    """
    if "'" not in value and "\n" not in value and "\r" not in value:
        return f"'{value}'"
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


@dataclass
class Config:
    values: dict[str, str] = field(default_factory=lambda: dict(DEFAULTS))
    path: Path = CONFIG_PATH
    just_created: bool = False

    # -- construction ---------------------------------------------------------------------
    @classmethod
    def load(cls, *, create: bool = False, path: Path | None = None) -> "Config":
        """Load config, layering the file (if any) over :data:`DEFAULTS`.

        With ``create=True`` and no file present, the defaults are written to disk and
        ``just_created`` is set so the caller can print the one-time setup hint. ``path``
        overrides the default location (used by tests).
        """
        target = path or CONFIG_PATH
        values = dict(DEFAULTS)
        if target.exists():
            try:
                with target.open("rb") as fh:
                    data = tomllib.load(fh)
            except (tomllib.TOMLDecodeError, OSError) as exc:
                raise ConfigError(
                    f"Could not read config at {target}: {exc}",
                    hint="Fix the file by hand or delete it to regenerate defaults.",
                ) from exc
            for key, raw in data.items():
                values[key] = str(raw)
            return cls(values=values, path=target, just_created=False)

        cfg = cls(values=values, path=target, just_created=False)
        if create:
            cfg.save()
            cfg.just_created = True
        return cfg

    # -- accessors ------------------------------------------------------------------------
    def raw(self, key: str) -> str:
        if key not in self.values and key not in DEFAULTS:
            raise ConfigError(f"Unknown config key: {key!r}")
        return self.values.get(key, DEFAULTS.get(key, ""))

    def expanded(self, key: str) -> Path:
        return expand_path(self.raw(key))

    def set(self, key: str, value: str) -> None:
        if key not in DEFAULTS:
            known = ", ".join(DEFAULTS)
            raise ConfigError(f"Unknown config key {key!r}. Known keys: {known}")
        self.values[key] = value

    # -- typed shortcuts ------------------------------------------------------------------
    @property
    def hf_home(self) -> Path:
        return self.expanded("hf_home")

    @property
    def gguf_dir(self) -> Path:
        return self.expanded("gguf_dir")

    @property
    def lmstudio_dir(self) -> Path:
        return self.expanded("lmstudio_dir")

    @property
    def ollama_models(self) -> Path:
        return self.expanded("ollama_models")

    @property
    def llamacpp_dir(self) -> Path:
        return self.expanded("llamacpp_dir")

    @property
    def llama_quantize(self) -> Path:
        return self.expanded("llama_quantize")

    @property
    def convert_script(self) -> Path:
        return self.llamacpp_dir / "convert_hf_to_gguf.py"

    @property
    def ollama_bin(self) -> str:
        return self.raw("ollama_bin")

    @property
    def default_quant(self) -> str:
        return self.raw("default_quant")

    @property
    def download_timeout(self) -> int:
        return _int_or(self.raw("download_timeout"), 30)

    @property
    def download_stall_timeout(self) -> int:
        return _int_or(self.raw("download_stall_timeout"), 300)

    # -- persistence ----------------------------------------------------------------------
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# mdl configuration",
            "# Paths use TOML literal strings (single quotes) so Windows backslashes are",
            "# NOT treated as escapes. %VARS% / $VARS and a leading ~ are expanded at runtime.",
            "",
        ]
        # known keys first, in canonical order, then any extras the user added
        for key in DEFAULTS:
            lines.append(f"{key} = {_dump_value(self.values.get(key, DEFAULTS[key]))}")
        for key, value in self.values.items():
            if key not in DEFAULTS:
                lines.append(f"{key} = {_dump_value(value)}")
        self.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
