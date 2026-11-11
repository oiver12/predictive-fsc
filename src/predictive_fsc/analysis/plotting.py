"""Render saved eligible incumbent traces without running a solver."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from ..io import load_result


def _label(result, path):
    name = result.provenance.build_identity.get("problem_name", Path(path).stem)
    config = result.config
    return f"{name} {config.initialization} M={config.memory} seed={config.seed}"


def plot_results(paths, output):
    """Plot each result's incumbent trace; return the paths that had nothing to plot."""
    if not paths:
        raise ValueError("no result files supplied")
    output = Path(output)
    if output.suffix.lower() not in (".svg", ".pdf", ".png"):
        raise ValueError("figure output must end in .svg, .pdf or .png")
    skipped = []
    figure, axes = plt.subplots(figsize=(7, 4), layout="constrained")
    try:
        for path in paths:
            result = load_result(path)
            cutoff = result.config.cpu_seconds
            rows = result.eligible
            if not rows:
                skipped.append(path)
                continue
            times = [row.cpu_seconds for row in rows]
            values = np.maximum.accumulate([row.value for row in rows])
            end = result.cpu_seconds if cutoff is None else cutoff
            # Repeat the final incumbent so the step curve extends to the end of the budget.
            axes.step([*times, end], [*values, values[-1]], where="post",
                      label=_label(result, path))
        axes.set(xlabel="CPU seconds", ylabel="Expected return", title="Completed eligible controllers")
        axes.grid(alpha=0.25)
        if axes.lines:
            axes.legend(fontsize="small")
        else:
            axes.text(0.5, 0.5, "No eligible candidates", ha="center", transform=axes.transAxes)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output)
    finally:
        plt.close(figure)
    return skipped
