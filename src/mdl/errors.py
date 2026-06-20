"""Friendly error types.

Anything that subclasses :class:`MdlError` is an *expected* failure (a missing drive, a
gated repo, a tool that isn't installed). The CLI catches these and prints a clean message
plus an optional hint -- never a raw traceback. Unexpected exceptions still propagate so
real bugs are visible (and shown in full under ``--verbose``).
"""

from __future__ import annotations


class MdlError(Exception):
    """Base class for expected, user-facing errors."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class ConfigError(MdlError):
    """Bad or missing configuration."""


class ToolNotFoundError(MdlError):
    """An external tool (ollama.exe, hf, llama-quantize.exe, ...) was not found."""


class DownloadError(MdlError):
    """A download failed."""


class GatedRepoError(MdlError):
    """The repo is gated/private and the user is not authenticated."""


class ConvertError(MdlError):
    """safetensors -> GGUF conversion or quantization failed."""


class RegistrationError(MdlError):
    """Wiring a model into a runtime (Ollama / LM Studio) failed."""
