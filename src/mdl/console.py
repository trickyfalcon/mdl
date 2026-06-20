"""Shared Rich console + small print helpers and global run flags."""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console

console = Console()
err_console = Console(stderr=True)


@dataclass
class RunFlags:
    """Process-wide flags set from the global CLI options."""

    dry_run: bool = False
    verbose: bool = False


flags = RunFlags()


def is_dry() -> bool:
    return flags.dry_run


def is_verbose() -> bool:
    return flags.verbose


def plan(message: str) -> None:
    """Print a planned action (shown instead of doing it under ``--dry-run``)."""
    console.print(f"[yellow]\\[dry-run][/] {message}")


def info(message: str) -> None:
    console.print(message)


def step(message: str) -> None:
    console.print(f"[cyan]->[/] {message}")


def success(message: str) -> None:
    console.print(f"[green]OK[/] {message}")


def warn(message: str) -> None:
    err_console.print(f"[yellow]warning:[/] {message}")
