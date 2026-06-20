"""Ollama integration.

Ollama can't read a loose GGUF: it imports via a Modelfile (``FROM <abs path>``) and
``ollama create`` *copies* the GGUF into its blob store. That copy is unavoidable; to keep it
on the fast disk we set ``OLLAMA_MODELS`` in the child env and ``doctor`` nudges the user to
set it globally (and on D:). The ``FROM`` path is double-quoted because Windows paths contain
backslashes and spaces.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from ..console import is_dry, plan, step, success
from ..errors import RegistrationError, ToolNotFoundError
from ..paths import split_repo_id
from ..proc import output_of, run


def build_modelfile(gguf_path: str | os.PathLike[str], *, extra_lines: list[str] | None = None) -> str:
    """Modelfile content importing a single GGUF by absolute path.

    The path is wrapped in double quotes so spaces and backslashes survive verbatim.
    """
    lines = [f'FROM "{Path(gguf_path)}"']
    if extra_lines:
        lines.extend(extra_lines)
    return "\n".join(lines) + "\n"


def model_name_for(repo_id: str, quant: str | None) -> str:
    """Derive a valid Ollama model name from a repo id (+ quant tag).

    ``bartowski/Qwen3-32B-GGUF`` + ``Q4_K_M`` -> ``qwen3-32b:q4_k_m``. Ollama names allow
    ``[a-z0-9._-]`` with an optional ``:tag``.
    """
    _owner, name = split_repo_id(repo_id)
    base = re.sub(r"(?i)[-_.]?gguf$", "", name)
    base = re.sub(r"[^a-z0-9._-]+", "-", base.lower()).strip("-._") or "model"
    if quant:
        tag = re.sub(r"[^a-z0-9._-]+", "-", quant.lower()).strip("-._")
        return f"{base}:{tag}" if tag else base
    return base


def _ollama_env(cfg) -> dict:
    env = dict(os.environ)
    env["OLLAMA_MODELS"] = str(cfg.ollama_models)
    return env


def ollama_running(cfg) -> tuple[bool, str]:
    """``(running, detail)`` -- talks to the daemon via ``ollama list``."""
    try:
        proc = run([cfg.ollama_bin, "list"], env=_ollama_env(cfg), dry=False, timeout=15)
    except FileNotFoundError:
        return False, "ollama not found on PATH"
    except Exception as exc:  # timeout etc.
        return False, str(exc)
    if proc is None or proc.returncode == 0:
        return True, "responding"
    lines = output_of(proc).strip().splitlines()
    return False, (lines[-1] if lines else "not responding")


def list_models(cfg) -> list[str]:
    """Names currently known to Ollama (first column of ``ollama list``)."""
    try:
        proc = run([cfg.ollama_bin, "list"], env=_ollama_env(cfg), dry=False, timeout=15)
    except Exception:
        return []
    if proc is None or proc.returncode != 0:
        return []
    names: list[str] = []
    for line in output_of(proc).splitlines()[1:]:  # skip header
        line = line.strip()
        if line:
            names.append(line.split()[0])
    return names


def model_exists(cfg, name: str) -> bool:
    return name in list_models(cfg)


def import_gguf(cfg, gguf_path: Path, name: str, *, force: bool = False) -> str:
    """Import a GGUF into Ollama as ``name`` (idempotent). Returns the model name."""
    if not force and not is_dry() and model_exists(cfg, name):
        step(f"ollama: '{name}' already imported -- skipping")
        return name

    content = build_modelfile(gguf_path)
    if is_dry():
        plan(f"write Modelfile:\n    {content.strip()}")
        plan(
            f"{cfg.ollama_bin} create {name} -f Modelfile   "
            f"(copies GGUF into Ollama blob store under {cfg.ollama_models})"
        )
        return name

    tmpdir = Path(tempfile.mkdtemp(prefix="mdl-ollama-"))
    modelfile = tmpdir / "Modelfile"
    modelfile.write_text(content, encoding="utf-8")
    # `ollama create` copies the whole GGUF into the blob store -- minutes for a big model.
    # Stream it (inherit the terminal) so its progress shows live instead of looking frozen.
    step(f"ollama: importing '{name}' -- copies the GGUF into Ollama's blob store (can take a while)")
    try:
        proc = run(
            [cfg.ollama_bin, "create", name, "-f", str(modelfile)],
            env=_ollama_env(cfg),
            label=f"{cfg.ollama_bin} create {name} -f Modelfile",
            stream=True,
        )
    except FileNotFoundError as exc:
        raise ToolNotFoundError(
            f"'{cfg.ollama_bin}' was not found.",
            hint="Install Ollama and ensure the `ollama` binary is on PATH, or set `ollama_bin` in config.",
        ) from exc
    finally:
        try:
            modelfile.unlink(missing_ok=True)
            tmpdir.rmdir()
        except OSError:
            pass

    if proc is not None and proc.returncode != 0:
        # streamed -> nothing captured; probe the daemon to explain the failure
        running, _ = ollama_running(cfg)
        if not running:
            raise RegistrationError(
                "Ollama service is not running.",
                hint="Start Ollama (run `ollama serve` or launch the app) and retry.",
            )
        raise RegistrationError(f"`ollama create {name}` failed (see the output above).")
    if proc is not None:
        success(f"ollama: imported as [bold]{name}[/]")
    return name


def remove(cfg, name: str) -> None:
    """``ollama rm <name>``. Missing models are not an error."""
    try:
        proc = run([cfg.ollama_bin, "rm", name], env=_ollama_env(cfg), label=f"{cfg.ollama_bin} rm {name}")
    except FileNotFoundError as exc:
        raise ToolNotFoundError(f"'{cfg.ollama_bin}' was not found.") from exc
    if proc is not None and proc.returncode != 0:
        out = output_of(proc).lower()
        if "not found" in out:
            step(f"ollama: '{name}' already absent")
            return
        raise RegistrationError(f"`ollama rm {name}` failed.\n{output_of(proc).strip()}".rstrip())
    if proc is not None:
        success(f"ollama: removed {name}")
