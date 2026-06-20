"""``mdl verify`` -- reconcile downloaded models against the Hub and optionally repair gaps.

For each format a model has (raw safetensors, per-quant GGUF) we compare what's on disk to
the Hub's file list/sizes and print a status line. With ``--repair`` we re-run the relevant
``hf download`` for any format that isn't complete -- ``hf`` resumes and fills only the
missing/mismatched files. Returns ``True`` only when every format is verified complete.
"""

from __future__ import annotations

from . import hub
from .console import console, info, success, warn
from .errors import MdlError
from .library import Library
from .paths import human_size, lmstudio_target_dir

_STATE_STYLE = {"complete": "green", "partial": "yellow", "missing": "red"}


def _format_status(label: str, s: "hub.DownloadStatus") -> str:
    if not s.verified:
        return f"  {label:<12} [yellow]present (unverified -- Hub unreachable)[/]  {human_size(s.present_bytes)} on disk"
    detail = human_size(s.present_bytes)
    if s.expected_bytes:
        pct = int(100 * s.present_bytes / s.expected_bytes)
        detail = f"{human_size(s.present_bytes)} / {human_size(s.expected_bytes)} ({pct}%)"
    extra = f", {s.incomplete} partial file(s)" if s.incomplete else ""
    style = _STATE_STYLE.get(s.state, "yellow")
    return f"  {label:<12} [{style}]{s.state}[/]  {detail}{extra}"


def _resolve(cfg, library: Library, query: str):
    """Return ``(name, raw_repo, gguf_repo, quants)`` for a library entry or a bare repo id."""
    rec = library.find(query)
    if rec is not None:
        return rec.model, rec.raw_repo, rec.gguf_repo, (rec.quants or [cfg.default_quant])
    if "/" in query:
        return query, query, None, []  # not tracked: treat as a raw repo id
    raise MdlError(
        f"'{query}' is not in the library and is not a repo id.",
        hint="Run `mdl list` to see tracked models, or pass owner/name.",
    )


def verify_model(cfg, library: Library, query: str, *, repair: bool = False, retries: int = 0) -> bool:
    name, raw_repo, gguf_repo, quants = _resolve(cfg, library, query)
    console.rule(f"[bold]verify[/] {name}")

    # (kind, ref, status) where ref is what the repairer needs to re-download
    checks: list[tuple[str, object, "hub.DownloadStatus"]] = []
    if raw_repo:
        s = hub.raw_status(cfg, raw_repo)
        console.print(_format_status("raw", s))
        checks.append(("raw", raw_repo, s))
    if gguf_repo:
        for q in quants:
            target = lmstudio_target_dir(cfg.gguf_dir, gguf_repo)
            s = hub.gguf_status(cfg, gguf_repo, q, target)
            console.print(_format_status(f"gguf {q}", s))
            checks.append(("gguf", (gguf_repo, q, target), s))
    if not checks:
        warn(f"{name}: nothing to verify (no raw or GGUF recorded).")
        return True

    incomplete = [(kind, ref) for kind, ref, s in checks if s.verified and s.state != "complete"]
    unverified = [kind for kind, _ref, s in checks if not s.verified]

    if repair and incomplete:
        console.print()
        for kind, ref in incomplete:
            if kind == "raw":
                info(f"repairing raw '{ref}' ...")
                hub.download_raw(cfg, ref, retries=retries)  # type: ignore[arg-type]
            else:
                grepo, q, target = ref  # type: ignore[misc]
                info(f"repairing gguf '{q}' from {grepo} ...")
                hub.download_gguf(cfg, grepo, q, target, retries=retries)
        console.print()
        return verify_model(cfg, library, query, repair=False, retries=0)  # re-check post-repair

    console.print()
    if not incomplete and not unverified:
        success(f"{name}: all formats complete.")
        return True
    if incomplete:
        tail = "" if repair else "  Re-run with --repair to fill the gaps."
        warn(f"{name}: {len(incomplete)} format(s) incomplete.{tail}")
    if unverified:
        warn(f"{name}: {len(unverified)} format(s) could not be checked (Hub unreachable).")
    return False
