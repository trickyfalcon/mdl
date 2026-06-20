"""LM Studio integration -- no symlinks, no copies.

The integration is "make the gguf disk *be* LM Studio's models directory". Because mdl
downloads GGUFs into ``<gguf_dir>\\<publisher>\\<model>\\file.gguf`` -- exactly LM Studio's
required layout -- LM Studio lists them with zero duplication once its models folder points at
``gguf_dir``. So mdl's job here is only to (a) verify a model sits in the correct
``publisher\\model`` structure, and (b) check LM Studio's configured folder and advise.

LM Studio stores its models directory in ``~/.lmstudio/settings.json`` under ``downloadsFolder``.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..console import is_dry, plan, success, warn
from ..paths import same_path

SETTINGS_PATH = Path.home() / ".lmstudio" / "settings.json"


def detect_models_dir() -> Path | None:
    """Read LM Studio's configured models folder, or ``None`` if it can't be determined."""
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    folder = data.get("downloadsFolder")
    return Path(folder) if folder else None


def verify_structure(gguf_dir: Path, target_dir: Path) -> tuple[bool, list[str]]:
    """Confirm ``target_dir`` is ``<gguf_dir>\\<publisher>\\<model>`` and holds a .gguf."""
    issues: list[str] = []
    gguf_dir = Path(gguf_dir)
    target_dir = Path(target_dir)
    try:
        rel = target_dir.relative_to(gguf_dir)
        if len(rel.parts) != 2:
            issues.append(
                f"{target_dir} is not a <publisher>\\<model> folder directly under {gguf_dir}"
            )
    except ValueError:
        issues.append(f"{target_dir} is not under gguf_dir ({gguf_dir})")
    if not (target_dir.exists() and any(target_dir.rglob("*.gguf"))):
        issues.append(f"no .gguf files found in {target_dir}")
    return (not issues), issues


def advise_models_dir(cfg) -> bool:
    """Compare LM Studio's folder with gguf_dir and advise. Returns True if they match.

    Uses the value LM Studio actually has in settings.json; if that can't be read, falls back
    to the configured ``lmstudio_dir`` (which exists for exactly this compare/advise purpose).
    """
    target = cfg.gguf_dir
    detected = detect_models_dir() or cfg.lmstudio_dir
    if same_path(detected, target):
        success(f"lmstudio: models folder already points at gguf_dir ({target})")
        return True
    warn(
        f"lmstudio: models folder is {detected}, not gguf_dir {target}.\n"
        f"          Fix in LM Studio: My Models > the folder path (top) > set to {target}\n"
        f'          (or set "downloadsFolder" in {SETTINGS_PATH} while LM Studio is closed).'
    )
    return False


def register(cfg, target_dir: Path) -> bool:
    """Verify structure and advise on the models folder. No file operations."""
    if is_dry():
        plan(f"verify {target_dir} holds the GGUF in publisher\\model layout for LM Studio")
        advise_models_dir(cfg)
        return True
    ok, issues = verify_structure(cfg.gguf_dir, target_dir)
    if ok:
        success(f"lmstudio: GGUF is in the correct publisher\\model layout ({target_dir})")
    else:
        for issue in issues:
            warn(f"lmstudio: {issue}")
    advise_models_dir(cfg)
    return ok
