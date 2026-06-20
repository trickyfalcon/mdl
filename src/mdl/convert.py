"""safetensors -> GGUF conversion + quantization.

Two steps, matching the spec ("run as ``python <llamacpp_dir>\\convert_hf_to_gguf.py ...``"):

1. ``convert_hf_to_gguf.py`` produces a high-precision GGUF (f16). With ``--remote`` it streams
   the weights straight from the Hub instead of needing a local snapshot.
2. ``llama-quantize.exe`` squeezes that down to the requested quant (e.g. Q4_K_M).

The convert script needs torch/transformers/gguf etc., which mdl deliberately does *not* carry.
We run it through ``uv run --with-requirements <llamacpp>/requirements/...`` so those heavy deps
live in an ephemeral environment instead of polluting mdl. The command builders are pure so they
can be unit tested and shown verbatim under ``--dry-run``.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from .console import is_dry, step, success, warn
from .errors import ConvertError, ToolNotFoundError
from .osenv import exe
from .proc import output_of, run

#: quant labels that are produced directly by the converter (no quantize step)
DIRECT_TYPES = {"F16", "F32", "BF16"}


def _uv_exe() -> Path | None:
    found = shutil.which("uv")
    if found:
        return Path(found)
    candidate = Path.home() / ".local" / "bin" / exe("uv")
    return candidate if candidate.exists() else None


def convert_requirements(cfg) -> Path:
    return cfg.llamacpp_dir / "requirements" / "requirements-convert_hf_to_gguf.txt"


def python_runner(cfg) -> list[str]:
    """Prefix that runs a Python able to import the converter's deps.

    Prefers ``uv run --with-requirements <llamacpp reqs>`` so torch/transformers are provided
    on demand; falls back to a curated ``--with`` set, then to mdl's own interpreter (which
    will only work if the user installed the deps themselves).
    """
    uv = _uv_exe()
    req = convert_requirements(cfg)
    if uv and req.exists():
        return [str(uv), "run", "--no-project", "--python", "3.12", "--with-requirements", str(req), "python"]
    if uv:
        extras = ["torch", "transformers", "sentencepiece", "gguf", "protobuf", "numpy", "safetensors"]
        cmd = [str(uv), "run", "--no-project", "--python", "3.12"]
        for pkg in extras:
            cmd += ["--with", pkg]
        return cmd + ["python"]
    return [sys.executable]


def convert_command(cfg, model_arg: str | Path, out_file: Path, *, remote: bool, outtype: str = "f16") -> list[str]:
    """``python convert_hf_to_gguf.py [--remote] --outfile <out> --outtype <t> <model>``."""
    cmd = python_runner(cfg) + [str(cfg.convert_script)]
    if remote:
        cmd.append("--remote")
    cmd += ["--outfile", str(out_file), "--outtype", outtype, str(model_arg)]
    return cmd


def quantize_command(cfg, in_file: Path, out_file: Path, quant: str) -> list[str]:
    """``llama-quantize.exe <in.gguf> <out.gguf> <QUANT>``."""
    return [str(cfg.llama_quantize), str(in_file), str(out_file), quant]


def ensure_tools(cfg, *, need_quantize: bool) -> None:
    if not cfg.convert_script.exists():
        raise ToolNotFoundError(
            f"convert_hf_to_gguf.py not found at {cfg.convert_script}.",
            hint="Set `llamacpp_dir` to your llama.cpp checkout (or clone it there).",
        )
    if need_quantize and not cfg.llama_quantize.exists():
        raise ToolNotFoundError(
            f"{exe('llama-quantize')} not found at {cfg.llama_quantize}.",
            hint="Build llama.cpp (cmake --build build --config Release) or set `llama_quantize` in config.",
        )


def convert_model(
    cfg,
    *,
    source: str | Path,
    quant: str,
    target_dir: Path,
    model_name: str,
    remote: bool = False,
) -> Path:
    """Convert ``source`` (local dir or, with ``remote``, a repo id) to a ``quant`` GGUF.

    Returns the path of the produced quantized file inside ``target_dir``.
    """
    quant_up = quant.upper()
    direct = quant_up in DIRECT_TYPES
    if is_dry():
        if not cfg.convert_script.exists():
            warn(f"convert script missing at {cfg.convert_script} (plan shown anyway)")
    else:
        ensure_tools(cfg, need_quantize=not direct)
        target_dir.mkdir(parents=True, exist_ok=True)

    final = target_dir / f"{model_name}-{quant_up}.gguf"

    # idempotent: a previously-built quant is reused rather than rebuilt
    if not is_dry() and final.exists():
        step(f"gguf already built ({final.name}) -- skipping convert")
        return final

    if direct:
        step(f"convert {source} -> {final.name} ({quant_up}, no quantize step)")
        proc = run(
            convert_command(cfg, source, final, remote=remote, outtype=quant.lower()),
            label=f"convert_hf_to_gguf.py --outtype {quant.lower()} -> {final}",
            stream=True,
        )
        _check(proc, "conversion")
        return final

    intermediate = target_dir / f"{model_name}.f16.gguf"
    step(f"convert {source} -> {intermediate.name} (f16)")
    proc = run(
        convert_command(cfg, source, intermediate, remote=remote, outtype="f16"),
        label=f"convert_hf_to_gguf.py --outtype f16 -> {intermediate}",
        stream=True,
    )
    _check(proc, "conversion")

    step(f"quantize {intermediate.name} -> {final.name} ({quant_up})")
    proc = run(
        quantize_command(cfg, intermediate, final, quant_up),
        label=f"llama-quantize {intermediate.name} {final.name} {quant_up}",
        stream=True,
    )
    _check(proc, "quantization")

    if not is_dry() and intermediate.exists() and intermediate != final:
        try:
            intermediate.unlink()
        except OSError:
            warn(f"could not remove intermediate {intermediate}")
    if not is_dry():
        success(f"converted -> {final}")
    return final


def _check(proc, what: str) -> None:
    if proc is not None and proc.returncode != 0:
        # streamed runs capture nothing -> the real output already scrolled past on screen
        tail = "\n".join(output_of(proc).strip().splitlines()[-8:])
        msg = f"{what} failed.\n{tail}" if tail else f"{what} failed. See the output above."
        raise ConvertError(msg.rstrip())
