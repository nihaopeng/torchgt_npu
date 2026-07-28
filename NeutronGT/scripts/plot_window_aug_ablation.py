#!/usr/bin/env python3
"""Plot cumulative window augmentation ablation accuracy curves."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


STAGES = ("no_extra", "hub_half", "hub_related")
STAGE_LABELS = {
    "no_extra": "non-overlap window",
    "hub_half": "hub-overlap window",
    "hub_related": "ours",
}
STAGE_COLORS = {
    "no_extra": "#7f7f7f",
    "hub_half": "#ff7f0e",
    "hub_related": "#1f77b4",
}
DATASET_FLAGS = {
    "arxiv": "ogbn-arxiv",
    "amazon": "AmazonProducts",
    "reddit": "reddit",
    "products": "ogbn-products",
}
DATASET_TITLES = {
    "ogbn-arxiv": "(a) OAV",
    "ogbn-products": "(b) OPT",
    "reddit": "(c) RDT",
    "AmazonProducts": "(d) AZ",
}

LOG_NAME_RE = re.compile(
    r"(?P<dataset>.+?)_(?P<model>GPH_Slim)_(?P<stage>no_extra|hub_half|hub_related)_"
    r"e(?P<epochs>\d+)_nparts(?P<nparts>\d+)_.*\.log$"
)
ACC_RE = re.compile(
    r"Epoch:\s*(?P<epoch>\d+),\s*"
    r"train_acc:\s*(?P<train>[-+]?\d+(?:\.\d+)?)%,\s*"
    r"valid_acc:\s*(?P<valid>[-+]?\d+(?:\.\d+)?)%,\s*"
    r"test_acc:\s*(?P<test>[-+]?\d+(?:\.\d+)?)%"
)


@dataclass
class AblationSeries:
    path: Path
    dataset: str
    model: str
    stage: str
    nparts: int
    accuracies: dict[int, tuple[float, float, float]]


def parse_acc(log_path: Path) -> dict[int, tuple[float, float, float]]:
    accuracies: dict[int, tuple[float, float, float]] = {}
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = ACC_RE.search(line)
            if match:
                accuracies[int(match.group("epoch"))] = (
                    float(match.group("train")),
                    float(match.group("valid")),
                    float(match.group("test")),
                )
    return accuracies


def load_log_folder(log_dir: Path) -> dict[str, dict[str, AblationSeries]]:
    grouped: dict[str, dict[str, AblationSeries]] = {}
    for log_path in sorted(log_dir.glob("*.log")):
        match = LOG_NAME_RE.match(log_path.name)
        if not match:
            continue
        accuracies = parse_acc(log_path)
        if not accuracies:
            continue

        dataset = match.group("dataset")
        stage = match.group("stage")
        grouped.setdefault(dataset, {})[stage] = AblationSeries(
            path=log_path,
            dataset=dataset,
            model=match.group("model"),
            stage=stage,
            nparts=int(match.group("nparts")),
            accuracies=accuracies,
        )
    return grouped


def accuracy_value(values: tuple[float, float, float], split: str) -> float:
    if split == "train":
        return values[0]
    if split == "valid":
        return values[1]
    if split == "test":
        return values[2]
    raise ValueError(f"Unsupported accuracy split: {split}")


def set_tight_ylim(ax, values: list[float]) -> None:
    if not values:
        return
    low = min(values)
    high = max(values)
    span = high - low
    pad = max(span * 0.12, 0.05)
    ax.set_ylim(low - pad, high + pad)


def setup_matplotlib():
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to draw figures. Run this script inside the "
            "NeutronGT training environment, or install it with `pip install matplotlib`."
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def draw_dataset_axes(
    ax,
    dataset: str,
    series_by_stage: dict[str, AblationSeries],
    acc_split: str,
    zoom_start: int | None,
    show_ylabel: bool = True,
) -> None:
    from matplotlib.ticker import MaxNLocator

    all_values: list[float] = []
    for stage in STAGES:
        series = series_by_stage.get(stage)
        if series is None:
            continue
        epochs = sorted(epoch for epoch in series.accuracies if zoom_start is None or epoch >= zoom_start)
        values = [accuracy_value(series.accuracies[epoch], acc_split) for epoch in epochs]
        all_values.extend(values)
        ax.plot(
            epochs,
            values,
            label=STAGE_LABELS[stage],
            color=STAGE_COLORS[stage],
            linewidth=1.9,
            marker="o",
            markersize=3.0,
        )

    ax.set_xlabel("Epoch", fontsize=11)
    if show_ylabel:
        ax.set_ylabel(f"{acc_split.capitalize()} Accuracy (%)", fontsize=11)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    if zoom_start is not None:
        set_tight_ylim(ax, all_values)
    ax.text(
        0.5,
        -0.26,
        DATASET_TITLES.get(dataset, dataset),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=17,
        fontweight="bold",
    )


def plot_dataset(
    dataset: str,
    series_by_stage: dict[str, AblationSeries],
    output_dir: Path,
    acc_split: str,
    zoom_start: int | None,
    dpi: int,
) -> Path:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    model_names = sorted({series.model for series in series_by_stage.values()})
    model_label = model_names[0] if len(model_names) == 1 else "/".join(model_names)
    draw_dataset_axes(ax, dataset, series_by_stage, acc_split, zoom_start, show_ylabel=True)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    zoom_suffix = f"_from{zoom_start}" if zoom_start is not None else ""
    output_path = output_dir / f"{dataset}_{model_label}_{acc_split}_ablation{zoom_suffix}.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_ab_pair(
    grouped: dict[str, dict[str, AblationSeries]],
    output_dir: Path,
    acc_split: str,
    zoom_start: int | None,
    dpi: int,
) -> Path | None:
    pair = ("ogbn-arxiv", "ogbn-products")
    if any(dataset not in grouped for dataset in pair):
        return None

    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    draw_dataset_axes(axes[0], pair[0], grouped[pair[0]], acc_split, zoom_start, show_ylabel=True)
    draw_dataset_axes(axes[1], pair[1], grouped[pair[1]], acc_split, zoom_start, show_ylabel=False)

    fig.tight_layout(w_pad=1.3, rect=(0, 0.10, 1, 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    zoom_suffix = f"_from{zoom_start}" if zoom_start is not None else ""
    output_path = output_dir / f"OAV_OPT_GPH_Slim_{acc_split}_ablation{zoom_suffix}.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_summary_csv(output_dir: Path, grouped: dict[str, dict[str, AblationSeries]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "window_aug_ablation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "model",
                "stage",
                "nparts",
                "acc_points",
                "final_acc_epoch",
                "final_train_acc",
                "final_valid_acc",
                "final_test_acc",
                "log_path",
            ]
        )
        for dataset in sorted(grouped):
            for stage in STAGES:
                series = grouped[dataset].get(stage)
                if series is None:
                    continue
                final_epoch = final_train = final_valid = final_test = ""
                if series.accuracies:
                    final_epoch = max(series.accuracies)
                    final_train, final_valid, final_test = series.accuracies[final_epoch]
                writer.writerow(
                    [
                        dataset,
                        series.model,
                        stage,
                        series.nparts,
                        len(series.accuracies),
                        final_epoch,
                        final_train,
                        final_valid,
                        final_test,
                        series.path,
                    ]
                )
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot no-extra, hub-half, and hub-related cumulative augmentation ablation accuracy curves."
    )
    parser.add_argument("log_dir", type=Path, help="Folder containing ablation *.log files.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <log_dir>/ablation_curves.",
    )
    parser.add_argument(
        "--acc-split",
        choices=("train", "valid", "test"),
        default="test",
        help="Accuracy split to plot. Default: test.",
    )
    parser.add_argument(
        "--zoom-start",
        type=int,
        default=None,
        help="Only plot points from this epoch onward and tighten the y-axis.",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Output image DPI. Default: 160.")
    parser.add_argument("--no-csv", action="store_true", help="Do not write summary CSV.")
    parser.add_argument("--arxiv", action="store_true", help="Only plot ogbn-arxiv logs.")
    parser.add_argument("--amazon", action="store_true", help="Only plot AmazonProducts logs.")
    parser.add_argument("--reddit", action="store_true", help="Only plot reddit logs.")
    parser.add_argument("--products", action="store_true", help="Only plot ogbn-products logs.")
    return parser.parse_args()


def selected_datasets(args: argparse.Namespace) -> set[str] | None:
    selected = {dataset for flag, dataset in DATASET_FLAGS.items() if getattr(args, flag)}
    return selected or None


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir.expanduser().resolve()
    if not log_dir.is_dir():
        raise NotADirectoryError(f"Log folder does not exist: {log_dir}")

    output_dir = (args.output_dir or (log_dir / "ablation_curves")).expanduser().resolve()
    grouped = load_log_folder(log_dir)
    if not grouped:
        raise ValueError(f"No matching ablation logs found in: {log_dir}")
    dataset_filter = selected_datasets(args)
    if dataset_filter is not None:
        grouped = {dataset: series for dataset, series in grouped.items() if dataset in dataset_filter}
        if not grouped:
            expected = ", ".join(sorted(dataset_filter))
            raise ValueError(f"No matching ablation logs found for selected dataset(s): {expected}")

    for dataset in sorted(grouped):
        missing = [stage for stage in STAGES if stage not in grouped[dataset]]
        if missing:
            print(f"Warning: {dataset} missing stages: {', '.join(missing)}")
        output_path = plot_dataset(dataset, grouped[dataset], output_dir, args.acc_split, args.zoom_start, args.dpi)
        print(f"Figure saved to: {output_path}")

    ab_output_path = plot_ab_pair(grouped, output_dir, args.acc_split, args.zoom_start, args.dpi)
    if ab_output_path is not None:
        print(f"Paired figure saved to: {ab_output_path}")

    if not args.no_csv:
        csv_path = write_summary_csv(output_dir, grouped)
        print(f"Summary CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
