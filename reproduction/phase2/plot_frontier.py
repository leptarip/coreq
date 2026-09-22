#!/usr/bin/env python3
# Copyright (c) 2026 278097159+leptarip@users.noreply.github.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate the four-panel paper-draft figure for the frontier-recall story."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from source.design_exploration.commons.design_space import SPACE_SPEC


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "results" / "phase2" / "frontier"
OUTPUT_DIR = REPO_ROOT / "results" / "phase2" / "figures"

AUDITED_COUNT = 11_896
EXPECTED_BASELINE_IDS = (
    7388, 7376, 7256, 7364, 7358, 20636, 6242, 2677, 5317, 1624, 832,
)
EXPECTED_AUGMENTED_IDS = (
    7388, 7376, 7256, 7364, 6499, 7310, 20636, 6242, 2677, 5317, 1624, 832,
)
EXPECTED_NEW_IDS = (6499, 7310)
EXPECTED_DISPLACED_IDS = (7358,)
RE_STUDY_IDS = (6499, 6242)

GREEN = "#1b9e77"
ORANGE = "#d95f02"
HIGHLIGHT = "black"
GREY = "#666666"
LIGHT_GREY = "#b8b8b8"
FONT_SIZE = 8
WIDTH = 7.16
FRONTIER_COLORS = (
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02",
    "#a6761d", "#666666", "#1f78b4", "#b15928", "#6a3d9a", "#33a02c",
)
PARAMETER_ORDER = (
    "prob", "gen_t", "unc_p", "unc_v",
    "fault", "packet_drop_rate", "network.delay_min", "network.delay_avg",
    "network.jit", "ego_ad_period",
)


def _csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _step_xy(rows: Sequence[Mapping[str, float]]) -> Tuple[List[float], List[float]]:
    ordered = sorted(rows, key=lambda row: (float(row["metric"]), float(row["cost"])))
    return (
        [float(row["metric"]) for row in ordered],
        [float(row["cost"]) for row in ordered],
    )


def load_data():
    """Load the two frontiers the figure draws.

    Both tables summarise the held-out trials. The per-episode values behind
    them sit beside them in the same folder, and
    ``tests/phase2/test_frontier_tables.py`` checks every published number
    against those episodes.
    """
    baseline = [
        {
            "design_id": int(row["design_id"]),
            "cost": float(row["cost"]),
            "metric": float(row["heldout_r240_metric_mean"]),
            "source": "audited-set frontier",
        }
        for row in _csv_rows(DATA / "baseline_frontier.csv")
    ]
    augmented = [
        {
            "design_id": int(row["design_id"]),
            "cost": float(row["cost"]),
            "metric": float(row["metric"]),
            "source": str(row["source"]),
        }
        for row in _csv_rows(DATA / "combined_safe_frontier.csv")
    ]
    if tuple(row["design_id"] for row in baseline) != EXPECTED_BASELINE_IDS:
        raise RuntimeError("the baseline frontier changed")
    if tuple(row["design_id"] for row in augmented) != EXPECTED_AUGMENTED_IDS:
        raise RuntimeError("the augmented frontier changed")

    mean_min_d = {
        int(row["design_id"]): float(row["mean_min_d"])
        for row in _csv_rows(DATA / "combined_safe_frontier.csv")
    }
    return baseline, augmented, mean_min_d


def _design_parameters(design_ids: Iterable[int]) -> pd.DataFrame:
    grid = np.asarray(SPACE_SPEC.build_grid(), dtype=float)
    frame = pd.DataFrame(
        [grid[int(design_id)] for design_id in design_ids],
        index=[str(int(design_id)) for design_id in design_ids],
        columns=SPACE_SPEC.keys,
    )
    return frame.loc[:, list(PARAMETER_ORDER)]


def _plot_heatmap(
    ax,
    design_ids: Sequence[int],
    highlight: Sequence[int] = (),
    *,
    annotation_fontsize: float = 5.2,
    tick_fontsize: float | None = None,
    colorbar_label: bool = True,
) -> None:
    raw = _design_parameters(design_ids)
    normalized = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    cmap = LinearSegmentedColormap.from_list("light_grey", ["#fafafa", "#969696"])
    image = ax.imshow(normalized.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    labels = {
        "prob": r"$R_d$", "gen_t": r"$R_t$", "fault": r"$N_f$",
        "unc_p": r"$R_p$", "unc_v": r"$R_v$", "ego_ad_period": r"$E_d$",
        "packet_drop_rate": r"$N_d$", "network.delay_min": r"$N_m$",
        "network.delay_avg": r"$N_a$", "network.jit": r"$N_j$",
    }
    ax.set_xticks(np.arange(raw.shape[1]))
    ax.set_xticklabels([labels.get(key, key) for key in raw.columns])
    ax.set_yticks(np.arange(raw.shape[0]))
    ax.set_yticklabels(["%s" % value for value in raw.index])
    if tick_fontsize is not None:
        ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.set_xlabel("Design parameters")
    ax.set_ylabel("Design ID")
    ax.axvline(3.5, color="black", linewidth=0.6)
    ax.axvline(8.5, color="black", linewidth=0.6)
    for row in range(raw.shape[0]):
        for column in range(raw.shape[1]):
            ax.text(
                column, row, "%g" % raw.iloc[row, column],
                ha="center", va="center", fontsize=annotation_fontsize,
            )
    highlighted = {str(int(design_id)) for design_id in highlight}
    for row, design_id in enumerate(raw.index):
        if design_id not in highlighted:
            continue
        ax.add_patch(Rectangle(
            (-0.5, row - 0.5), raw.shape[1], 1.0, fill=False,
            edgecolor=HIGHLIGHT, linewidth=1.5, zorder=5,
        ))
        ax.get_yticklabels()[row].set_color(HIGHLIGHT)
        ax.get_yticklabels()[row].set_fontweight("bold")
    colorbar = plt.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    if colorbar_label:
        colorbar.set_label("Normalized value")
    if tick_fontsize is not None:
        colorbar.ax.tick_params(labelsize=tick_fontsize)


DEFAULT_LABEL_OFFSETS = {
    6499: (13, -19), 7310: (31, -15), 7358: (11, 15), 7256: (4, 20), 6242: (0, 20),
}
COMPACT_LABEL_OFFSETS = {
    6499: (11, -15), 7310: (16, -13), 7358: (0, 14), 7256: (3, 15), 6242: (0, 15),
}


def _draw_extension_panel(
    ax, baseline, augmented, selected, *,
    scale: float = 1.0,
    legend_fontsize: float = 6.3,
    label_fontsize: float = 6.7,
    legend_markersize: float | None = None,
    label_offsets: Mapping[int, Tuple[float, float]] = DEFAULT_LABEL_OFFSETS,
) -> None:
    """Panel (b): the recovered designs entering the frontier, with the study pair marked."""
    old_x, old_y = _step_xy(baseline)
    new_x, new_y = _step_xy(augmented)
    ax.plot(old_x, old_y, drawstyle="steps-post", color=LIGHT_GREY,
            linestyle="--", linewidth=1.1)
    ax.plot(new_x, new_y, drawstyle="steps-post", color=GREY, linewidth=1.35)
    retained = [row for row in baseline if row["design_id"] not in EXPECTED_DISPLACED_IDS]
    displaced = [row for row in baseline if row["design_id"] in EXPECTED_DISPLACED_IDS]
    added = [row for row in augmented if row["design_id"] in EXPECTED_NEW_IDS]
    ax.scatter(
        [row["metric"] for row in retained], [row["cost"] for row in retained],
        marker="D", c=GREEN, edgecolors="black", linewidths=0.4, s=31 * scale, zorder=3,
    )
    ax.scatter(
        [row["metric"] for row in displaced], [row["cost"] for row in displaced],
        marker="D", facecolors="none", edgecolors=GREY, linewidths=1.2,
        s=40 * scale, zorder=3,
    )
    ax.scatter(
        [row["metric"] for row in added], [row["cost"] for row in added],
        marker="^", c=ORANGE, edgecolors="black", linewidths=0.55, s=64 * scale, zorder=5,
    )
    selected_rows = [row for row in augmented if row["design_id"] in selected]
    ax.scatter(
        [row["metric"] for row in selected_rows],
        [row["cost"] for row in selected_rows],
        marker="o", facecolors="none", edgecolors=HIGHLIGHT,
        linewidths=1.6, s=105 * scale, zorder=6,
    )
    tagged = [(row, ORANGE) for row in added] + [(row, GREY) for row in displaced]
    tagged_ids = {row["design_id"] for row, _ in tagged}
    tagged += [
        (row, HIGHLIGHT) for row in selected_rows
        if row["design_id"] not in tagged_ids
    ]
    for row, color in tagged:
        ax.annotate(
            str(row["design_id"]), (row["metric"], row["cost"]),
            textcoords="offset points", xytext=label_offsets[row["design_id"]],
            fontsize=label_fontsize,
            color=color, ha="center", va="center", zorder=7,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                      edgecolor=color, linewidth=0.55),
            arrowprops=dict(arrowstyle="-", color=color, linewidth=0.55,
                            shrinkA=1.5, shrinkB=3.0),
        )
    # Legend markers keep the rcParams size unless asked otherwise, so a shrunken
    # legend font does not leave the symbols overlapping their neighbouring rows.
    marker_size = {} if legend_markersize is None else {"markersize": legend_markersize}
    ax.legend(handles=[
        Line2D([], [], color=LIGHT_GREY, linestyle="--", label=r"$\mathcal{D}_{\mathrm{nd}}$"),
        Line2D([], [], color=GREY, label=r"$\mathcal{D}_{\mathrm{nd,ext}}$"),
        Line2D([], [], marker="^", linestyle="none", markerfacecolor=ORANGE,
               markeredgecolor="black", label="Recovered design", **marker_size),
        Line2D([], [], marker="D", linestyle="none", markerfacecolor="none",
               markeredgecolor=GREY, label="Displaced design", **marker_size),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor=HIGHLIGHT, label=r"$\mathcal{D}_{*}$ selected designs",
               **marker_size),
    ], loc="upper left", fontsize=legend_fontsize, frameon=True,
        labelspacing=0.4, handletextpad=0.5, borderpad=0.32)


def _apply_frontier_limits(axes: Sequence, baseline, augmented) -> None:
    all_metrics = [row["metric"] for row in baseline + augmented]
    all_costs = [row["cost"] for row in baseline + augmented]
    metric_pad = 0.045 * (max(all_metrics) - min(all_metrics))
    cost_pad = 0.055 * (max(all_costs) - min(all_costs))
    for frontier_ax in axes:
        frontier_ax.set_xlim(min(all_metrics) - metric_pad, max(all_metrics) + metric_pad)
        frontier_ax.set_ylim(min(all_costs) - cost_pad, max(all_costs) + cost_pad)
        frontier_ax.set_xlabel(r"Aggregated velocity at $d_{\min}$ [m/s]")
        frontier_ax.set_ylabel("Cost")
        frontier_ax.grid(True, linestyle=":", linewidth=0.55, alpha=0.55)


def _draw_metrics_panel(
    ax, augmented, augmented_mean_min_d, selected, *,
    legend_fontsize: float = 5.3,
    legend_ncol: int = 4,
    ylim: float = 1.23,
) -> None:
    """Panel (c): per-design metrics, min-max normalized with direction retained."""
    metric_matrix = np.asarray([
        [augmented_mean_min_d[row["design_id"]], row["metric"], row["cost"]]
        for row in augmented
    ], dtype=float)
    normalized = np.empty_like(metric_matrix)
    for column in range(metric_matrix.shape[1]):
        values = metric_matrix[:, column]
        normalized[:, column] = (
            (values - values.min()) / (values.max() - values.min() + 1e-9)
        )
    normalized = np.clip(normalized, 0.025, 1.0)
    x_positions = np.arange(3)
    width = 0.78 / len(augmented)
    for index, row in enumerate(augmented):
        is_selected = row["design_id"] in selected
        ax.bar(
            x_positions - 0.39 + (index + 0.5) * width,
            normalized[index], width=width,
            color=FRONTIER_COLORS[index], edgecolor=HIGHLIGHT,
            linewidth=1.5 if is_selected else 0.25,
            zorder=3 if is_selected else 2,
            label=str(row["design_id"]),
        )
    ax.set_xticks(x_positions)
    # Both KPIs are means over the confirmation episodes, so both carry the bar.
    ax.set_xticklabels([r"$\overline{d}_{\min}$", r"$\overline{v}@d_{\min}$", "Cost"])
    ax.set_ylim(0, ylim)
    ax.set_ylabel("Normalized value")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    legend = ax.legend(
        loc="upper center", ncol=legend_ncol, fontsize=legend_fontsize, frameon=True,
        framealpha=1.0, columnspacing=0.45, handletextpad=0.22, borderpad=0.3,
    )
    selected_labels = {str(design_id) for design_id in selected}
    for text in legend.get_texts():
        if text.get_text() in selected_labels:
            text.set_fontweight("bold")


def _study_designs(selected: Sequence[int]) -> List[int]:
    missing = [
        design_id for design_id in selected
        if design_id not in EXPECTED_AUGMENTED_IDS
    ]
    if missing:
        raise RuntimeError("rare-event study designs are not on the frontier: %s" % missing)
    return list(selected)


def build_three_panel_figure(
    baseline, augmented, augmented_mean_min_d,
    annotated: bool = True,
    panel_letters: Sequence[str] = ("a", "b", "c"),
) -> Tuple[plt.Figure, List[int]]:
    """Build the single-row variant holding the extension, metric and parameter panels.

    The width is the IEEE double-column text width, so the row fits a full-width
    figure environment without scaling; ``panel_letters`` relabels the panels for
    papers that keep the four-panel (b)-(d) lettering.
    """
    selected = _study_designs(RE_STUDY_IDS)

    fig, axes = plt.subplots(
        1, 3, figsize=(WIDTH, 2.62), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.36]},
    )
    augmented_ax, trade_ax, heat_ax = axes
    letters = list(panel_letters)

    def set_title(ax, index: int, full: str) -> None:
        ax.set_title(full if annotated else "(%s)" % letters[index], fontsize=7.5)

    _draw_extension_panel(
        augmented_ax, baseline, augmented, selected,
        scale=0.8, legend_fontsize=5.6, label_fontsize=5.6, legend_markersize=3.8,
        label_offsets=COMPACT_LABEL_OFFSETS,
    )
    set_title(
        augmented_ax, 0,
        r"(%s) $\mathcal{D}_{\mathrm{nd}}\rightarrow\mathcal{D}_{\mathrm{nd,ext}}$"
        % letters[0],
    )
    _apply_frontier_limits((augmented_ax,), baseline, augmented)
    augmented_ax.tick_params(axis="both", labelsize=6.0)
    augmented_ax.xaxis.label.set_size(6.8)
    augmented_ax.yaxis.label.set_size(6.8)

    _draw_metrics_panel(
        trade_ax, augmented, augmented_mean_min_d, selected,
        legend_fontsize=4.3, legend_ncol=4, ylim=1.46,
    )
    set_title(
        trade_ax, 1,
        r"(%s) Metrics of $\mathcal{D}_{\mathrm{nd,ext}}$" % letters[1],
    )
    trade_ax.tick_params(axis="both", labelsize=6.0)
    trade_ax.yaxis.label.set_size(6.8)

    _plot_heatmap(
        heat_ax, [row["design_id"] for row in augmented], highlight=selected,
        annotation_fontsize=4.0, tick_fontsize=5.6, colorbar_label=False,
    )
    set_title(
        heat_ax, 2,
        r"(%s) Parameters of $\mathcal{D}_{\mathrm{nd,ext}}$" % letters[2],
    )
    heat_ax.xaxis.label.set_size(6.8)
    heat_ax.yaxis.label.set_size(6.8)
    return fig, selected


def main() -> None:
    global DATA, OUTPUT_DIR
    import argparse
    parser = argparse.ArgumentParser(description="Plot the Phase 2 cost-performance frontier.")
    parser.add_argument("--data-dir", type=Path, default=DATA, help="Folder with the frontier CSV tables.")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="Folder for the figures.")
    args = parser.parse_args()
    DATA, OUTPUT_DIR = args.data_dir, args.out_dir
    plt.rcParams.update({
        "font.size": FONT_SIZE, "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE, "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE, "legend.fontsize": FONT_SIZE,
    })
    baseline, augmented, augmented_mean_min_d = load_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    figure, selected = build_three_panel_figure(
        baseline, augmented, augmented_mean_min_d, annotated=False,
    )
    path = OUTPUT_DIR / "frontier.svg"
    figure.savefig(path)
    written.append(str(path))
    plt.close(figure)
    print(json.dumps({
        "audited_design_count": AUDITED_COUNT,
        "baseline_frontier_ids": list(EXPECTED_BASELINE_IDS),
        "augmented_frontier_ids": list(EXPECTED_AUGMENTED_IDS),
        "rare_event_study_ids": selected,
        "outputs": written,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
