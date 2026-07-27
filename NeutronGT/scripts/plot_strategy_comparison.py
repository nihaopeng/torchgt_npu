#!/usr/bin/env python3
"""Plot vertex-copy strategy comparison curves from NeutronGT log folders."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


STRATEGIES = ("ours", "hub", "random", "related")
STRATEGY_COLORS = {
    "ours": "#1f77b4",
    "hub": "#ff7f0e",
    "random": "#7f7f7f",
    "related": "#2ca02c",
}

LOG_NAME_RE = re.compile(
    r"(?P<dataset>.+?)_(?P<model>GPH_Slim|GPH_Large|GT)_(?P<strategy>ours|hub|random|related)_"
    r"e(?P<epochs>\d+)_nparts(?P<nparts>\d+)_.*\.log$"
)
LOSS_RE = re.compile(
    r"Epoch:\s*(?P<epoch>\d+),\s*Loss:\s*(?P<loss>[-+]?\d+(?:\.\d+)?)"
)
ACC_RE = re.compile(
    r"Epoch:\s*(?P<epoch>\d+),\s*"
    r"train_acc:\s*(?P<train>[-+]?\d+(?:\.\d+)?)%,\s*"
    r"valid_acc:\s*(?P<valid>[-+]?\d+(?:\.\d+)?)%,\s*"
    r"test_acc:\s*(?P<test>[-+]?\d+(?:\.\d+)?)%"
)


@dataclass
class LogSeries:
    path: Path
    dataset: str
    model: str
    strategy: str
    nparts: int
    losses: dict[int, float]
    accuracies: dict[int, tuple[float, float, float]]


def parse_log(log_path: Path) -> tuple[dict[int, float], dict[int, tuple[float, float, float]]]:
    losses: dict[int, float] = {}
    accuracies: dict[int, tuple[float, float, float]] = {}

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            loss_match = LOSS_RE.search(line)
            if loss_match:
                losses[int(loss_match.group("epoch"))] = float(loss_match.group("loss"))
                continue

            acc_match = ACC_RE.search(line)
            if acc_match:
                accuracies[int(acc_match.group("epoch"))] = (
                    float(acc_match.group("train")),
                    float(acc_match.group("valid")),
                    float(acc_match.group("test")),
                )

    return losses, accuracies


def load_log_folder(log_dir: Path) -> dict[str, dict[str, LogSeries]]:
    grouped: dict[str, dict[str, LogSeries]] = {}
    for log_path in sorted(log_dir.glob("*.log")):
        match = LOG_NAME_RE.match(log_path.name)
        if not match:
            continue

        dataset = match.group("dataset")
        model = match.group("model")
        strategy = match.group("strategy")
        nparts = int(match.group("nparts"))
        losses, accuracies = parse_log(log_path)
        if not losses and not accuracies:
            continue

        grouped.setdefault(dataset, {})[strategy] = LogSeries(
            path=log_path,
            dataset=dataset,
            model=model,
            strategy=strategy,
            nparts=nparts,
            losses=losses,
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


def filter_points(
    values: dict[int, float],
    min_epoch: int | None,
) -> tuple[list[int], list[float]]:
    epochs = sorted(epoch for epoch in values if min_epoch is None or epoch >= min_epoch)
    return epochs, [values[epoch] for epoch in epochs]


def set_zoom_ylim(ax, values: list[float], min_pad: float) -> None:
    if not values:
        return
    low = min(values)
    high = max(values)
    span = high - low
    pad = max(span * 0.12, min_pad)
    ax.set_ylim(low - pad, high + pad)


def plot_dataset(
    dataset: str,
    series_by_strategy: dict[str, LogSeries],
    output_dir: Path,
    acc_split: str,
    zoom_start: int,
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

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    model_names = sorted({series.model for series in series_by_strategy.values()})
    nparts_values = sorted({series.nparts for series in series_by_strategy.values()})
    model_label = model_names[0] if len(model_names) == 1 else "/".join(model_names)
    nparts_label = str(nparts_values[0]) if len(nparts_values) == 1 else "/".join(map(str, nparts_values))
    zoom_loss_values: list[float] = []
    zoom_acc_values: list[float] = []

    for strategy in STRATEGIES:
        series = series_by_strategy.get(strategy)
        if series is None:
            continue

        color = STRATEGY_COLORS[strategy]
        if series.losses:
            loss_epochs = sorted(series.losses)
            loss_values = [series.losses[e] for e in loss_epochs]
            axes[0, 0].plot(
                loss_epochs,
                loss_values,
                label=strategy,
                color=color,
                linewidth=1.8,
            )
            zoom_epochs, zoom_values = filter_points(series.losses, zoom_start)
            zoom_loss_values.extend(zoom_values)
            axes[0, 1].plot(
                zoom_epochs,
                zoom_values,
                label=strategy,
                color=color,
                linewidth=1.8,
            )

        if series.accuracies:
            acc_epochs = sorted(series.accuracies)
            acc_values = [accuracy_value(series.accuracies[e], acc_split) for e in acc_epochs]
            axes[1, 0].plot(
                acc_epochs,
                acc_values,
                label=strategy,
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=3,
            )
            zoom_acc_epochs = [epoch for epoch in acc_epochs if epoch >= zoom_start]
            zoom_acc = [accuracy_value(series.accuracies[e], acc_split) for e in zoom_acc_epochs]
            zoom_acc_values.extend(zoom_acc)
            axes[1, 1].plot(
                zoom_acc_epochs,
                zoom_acc,
                label=strategy,
                color=color,
                linewidth=1.8,
                marker="o",
                markersize=3,
            )

    axes[0, 0].set_title("Loss")
    axes[0, 1].set_title(f"Loss after epoch {zoom_start}")
    axes[1, 0].set_title(f"{acc_split.capitalize()} Accuracy")
    axes[1, 1].set_title(f"{acc_split.capitalize()} Accuracy after epoch {zoom_start}")
    axes[0, 0].set_ylabel("Loss")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 1].set_xlabel("Epoch")
    set_zoom_ylim(axes[0, 1], zoom_loss_values, min_pad=0.01)
    set_zoom_ylim(axes[1, 1], zoom_acc_values, min_pad=0.05)

    for ax in axes.ravel():
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
        ax.legend(loc="best")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    fig.suptitle(f"{dataset} {model_label} vertex-copy strategies (n_parts={nparts_label})")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset}_{model_label}_{acc_split}_strategy_curves.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_summary_csv(output_dir: Path, grouped: dict[str, dict[str, LogSeries]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "strategy_curve_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "model",
                "strategy",
                "nparts",
                "loss_points",
                "acc_points",
                "final_loss_epoch",
                "final_loss",
                "final_acc_epoch",
                "final_train_acc",
                "final_valid_acc",
                "final_test_acc",
                "log_path",
            ]
        )
        for dataset in sorted(grouped):
            for strategy in STRATEGIES:
                series = grouped[dataset].get(strategy)
                if series is None:
                    continue
                final_loss_epoch = final_loss = ""
                if series.losses:
                    final_loss_epoch = max(series.losses)
                    final_loss = series.losses[final_loss_epoch]
                final_acc_epoch = final_train = final_valid = final_test = ""
                if series.accuracies:
                    final_acc_epoch = max(series.accuracies)
                    final_train, final_valid, final_test = series.accuracies[final_acc_epoch]
                writer.writerow(
                    [
                        dataset,
                        series.model,
                        strategy,
                        series.nparts,
                        len(series.losses),
                        len(series.accuracies),
                        final_loss_epoch,
                        final_loss,
                        final_acc_epoch,
                        final_train,
                        final_valid,
                        final_test,
                        series.path,
                    ]
                )
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot vertex-copy strategy comparison curves from a folder containing "
            "four NeutronGT strategy logs per dataset."
        )
    )
    parser.add_argument("log_dir", type=Path, help="Folder containing comparison *.log files.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <log_dir>/strategy_curves.",
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
        default=100,
        help="Start epoch for zoomed loss/accuracy panels. Default: 100.",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Output image DPI. Default: 160.")
    parser.add_argument("--no-csv", action="store_true", help="Do not write summary CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir.expanduser().resolve()
    if not log_dir.is_dir():
        raise NotADirectoryError(f"Log folder does not exist: {log_dir}")

    output_dir = (args.output_dir or (log_dir / "strategy_curves")).expanduser().resolve()
    grouped = load_log_folder(log_dir)
    if not grouped:
        raise ValueError(f"No matching strategy logs found in: {log_dir}")

    for dataset in sorted(grouped):
        missing = [strategy for strategy in STRATEGIES if strategy not in grouped[dataset]]
        if missing:
            print(f"Warning: {dataset} missing strategies: {', '.join(missing)}")
        output_path = plot_dataset(
            dataset,
            grouped[dataset],
            output_dir,
            args.acc_split,
            args.zoom_start,
            args.dpi,
        )
        print(f"Figure saved to: {output_path}")

    if not args.no_csv:
        csv_path = write_summary_csv(output_dir, grouped)
        print(f"Summary CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
