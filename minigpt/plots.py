"""The three or four charts we keep coming back to.

Everything here is plain matplotlib with big fonts and clear labels, because a
chart that is unreadable on a stream is a chart that taught nobody anything.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt

from minigpt.train import History

#: A colour per curve type, kept consistent across every notebook.
TRAIN_COLOR = "#1f77b4"
VAL_COLOR = "#d62728"


def use_stream_style() -> None:
    """Bump font sizes and figure size so charts read well on a stream."""
    plt.rcParams.update(
        {
            "figure.figsize": (9, 5),
            "figure.dpi": 110,
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "lines.linewidth": 2.2,
            "lines.markersize": 5,
        }
    )


def plot_curves(
    history: History,
    title: str | None = None,
    ax: plt.Axes | None = None,
    annotate_best: bool = True,
    random_baseline: float | None = None,
) -> plt.Axes:
    """Train loss and validation loss on one chart - the core diagnostic view."""
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(history.steps, history.train_loss, "-o", color=TRAIN_COLOR,
            label="train loss (the homework)")
    ax.plot(history.steps, history.val_loss, "-o", color=VAL_COLOR,
            label="validation loss (the pop quiz)")
    if random_baseline:
        ax.axhline(random_baseline, ls=":", color="grey",
                   label=f"random guessing ({random_baseline:.2f})")
    if annotate_best and history.val_loss:
        ax.axvline(history.best_val_step, ls="--", color="grey", alpha=0.7)
        ax.annotate(
            f"best val {history.best_val:.3f}\n@ step {history.best_val_step}",
            xy=(history.best_val_step, history.best_val),
            xytext=(8, 14),
            textcoords="offset points",
            fontsize=10,
            color="dimgrey",
        )
    ax.set_xlabel("training step")
    ax.set_ylabel("loss (lower is better)")
    ax.set_title(title or history.name)
    ax.legend()
    return ax


def plot_comparison(
    histories: Sequence[History],
    labels: Sequence[str] | None = None,
    title: str = "Three runs, three diagnoses",
) -> plt.Figure:
    """One panel per run, sharing a y-axis so the shapes are comparable."""
    labels = list(labels or [h.name for h in histories])
    fig, axes = plt.subplots(1, len(histories), figsize=(6 * len(histories), 4.6), sharey=True)
    if len(histories) == 1:
        axes = [axes]
    for ax, history, label in zip(axes, histories, labels):
        plot_curves(history, title=label, ax=ax, annotate_best=False)
        ax.legend(fontsize=9)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    return fig


def plot_two_domains(
    steps: Iterable[int],
    old_domain: Iterable[float],
    new_domain: Iterable[float],
    title: str = "Learning the new domain, forgetting the old one",
    old_label: str = "OLD domain val loss (stories)",
    new_label: str = "NEW domain val loss (recipes)",
) -> plt.Axes:
    """The catastrophic-forgetting chart from notebook 04."""
    _, ax = plt.subplots()
    steps = list(steps)
    ax.plot(steps, list(old_domain), "-o", color="#9467bd", label=old_label)
    ax.plot(steps, list(new_domain), "-o", color="#2ca02c", label=new_label)
    ax.set_xlabel("continued pre-training step")
    ax.set_ylabel("loss (lower is better)")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_bars(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    ylabel: str = "loss",
    colors: Sequence[str] | None = None,
) -> plt.Axes:
    """A simple before/after bar chart."""
    _, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(list(labels), list(values), color=list(colors) if colors else None)
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=11,
        )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax


__all__ = [
    "TRAIN_COLOR",
    "VAL_COLOR",
    "plot_bars",
    "plot_comparison",
    "plot_curves",
    "plot_two_domains",
    "use_stream_style",
]
