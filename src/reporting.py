"""Console output and file-saving helpers shared by the modelling modules."""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

import config as cfg

console = Console()

_STEP = {"n": 0, "total": 0}


def set_total_steps(n: int):
    _STEP.update(n=0, total=n)


@contextmanager
def step(title: str):
    """Numbered section header with elapsed time."""
    _STEP["n"] += 1
    console.print(f"\n[bold cyan]Stage {_STEP['n']}/{_STEP['total']} · {title}[/bold cyan]")
    t0 = time.perf_counter()
    yield
    console.print(f"[green]✓ done[/green] [dim]({time.perf_counter() - t0:.1f}s)[/dim]\n")


@contextmanager
def task(msg: str):
    """Spinner for a long-running sub-task."""
    t0 = time.perf_counter()
    with console.status(f"{msg}…"):
        yield
    console.print(f"  • {msg} [dim]({time.perf_counter() - t0:.1f}s)[/dim]")


def info(msg: str):
    console.print(f"  [dim]→[/dim] {msg}")


def warn(msg: str):
    console.print(f"  [yellow]⚠ {msg}[/yellow]")


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(cfg.ROOT))
    except ValueError:
        return str(path)


def save_csv(df: pd.DataFrame, path: Path, index: bool = False, quiet: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8-sig")  # BOM so Excel shows – and λ correctly
    if not quiet:
        info(f"saved [bold]{_rel(path)}[/bold]  {df.shape}")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def save_json(obj, path: Path, quiet: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    if not quiet:
        info(f"saved [bold]{_rel(path)}[/bold]")


def stars(p: float) -> str:
    """Significance stars used in the paper: * p<.10, ** p<.05, *** p<.01."""
    if pd.isna(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def label(col: str) -> str:
    name = cfg.FEATURE_LABELS.get(col)
    return f"{name} ({col})" if name else col


def markdown_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Minimal GitHub-flavoured markdown table (no extra dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("" if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
