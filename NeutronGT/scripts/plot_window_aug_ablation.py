#!/usr/bin/env python3
"""Plot cumulative window augmentation ablation accuracy curves."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


STAGES = ("hub_half", "hub_related")
STAGE_LABELS = {
    "hub_half": "50% hub",
    "hub_related": "50% hub + 50% related",
}
STAGE_COLORS = {
    "hub_half": "#ff7f0e",
    "hub_related": "#1f77b4",
}

LOG_NAME_RE = re.compile(
    r"(?P<dataset>.+?)_(?P<model>GPH_Slim)_(?P<stage>hub_half|hub_related)_"
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


def plot_dataset(
    dataset: str,
    series_by_stage: dict[str, AblationSeries],
    output_dir: Path,
    acc_split: str,
    zoom_start: int | None,
    dpi: int,
) -> Path:
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to draw figures. Run this script inside the "
            "NeutronGT training environment, or install it with `pip install matplotlib`."
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    all_values: list[float] = []
    model_names = sorted({series.model for series in series_by_stage.values()})
    nparts_values = sorted({series.nparts for series in series_by_stage.values()})
    model_label = model_names[0] if len(model_names) == 1 else "/".join(model_names)
    nparts_label = str(nparts_values[0]) if len(nparts_values) == 1 else "/".join(map(str, nparts_values))

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
            linewidth=2.0,
            marker="o",
            markersize=3.5,
        )

    ax.set_title(f"{dataset} {model_label} cumulative augmentation ablation (n_parts={nparts_label})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{acc_split.capitalize()} Accuracy (%)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.legend(loc="best")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    if zoom_start is not None:
        set_tight_ylim(ax, all_values)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    zoom_suffix = f"_from{zoom_start}" if zoom_start is not None else ""
    output_path = output_dir / f"{dataset}_{model_label}_{acc_split}_ablation{zoom_suffix}.png"
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
        description="Plot hub-half vs hub-related cumulative augmentation ablation accuracy curves."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir.expanduser().resolve()
    if not log_dir.is_dir():
        raise NotADirectoryError(f"Log folder does not exist: {log_dir}")

    output_dir = (args.output_dir or (log_dir / "ablation_curves")).expanduser().resolve()
    grouped = load_log_folder(log_dir)
    if not grouped:
        raise ValueError(f"No matching ablation logs found in: {log_dir}")

    for dataset in sorted(grouped):
        missing = [stage for stage in STAGES if stage not in grouped[dataset]]
        if missing:
            print(f"Warning: {dataset} missing stages: {', '.join(missing)}")
        output_path = plot_dataset(dataset, grouped[dataset], output_dir, args.acc_split, args.zoom_start, args.dpi)
        print(f"Figure saved to: {output_path}")

    if not args.no_csv:
        csv_path = write_summary_csv(output_dir, grouped)
        print(f"Summary CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
