#!/usr/bin/env python3
"""Plot precision comparison charts from runs/*/summary.csv and result_*.json."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


ORDER = ["fp32", "int8", "int4", "int2", "bit158"]
COLORS = {
    "fp32": "#2563eb",
    "int8": "#059669",
    "int4": "#d97706",
    "int2": "#dc2626",
    "bit158": "#7c3aed",
}
LABELS = {
    "fp32": "FP32",
    "int8": "INT8",
    "int4": "INT4",
    "int2": "INT2",
    "bit158": "Bit1.58",
}


def load_summary(run_dir: Path):
    rows = {}
    with open(run_dir / "summary.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["precision"]] = r
    return rows


def load_histories(run_dir: Path):
    hist = {}
    for p in ORDER:
        path = run_dir / f"result_{p}.json"
        if path.exists():
            hist[p] = json.loads(path.read_text(encoding="utf-8"))
    return hist


def plot_bar(rows, out_path: Path):
    precisions = [p for p in ORDER if p in rows]
    accs = [float(rows[p]["best_val_acc"]) for p in precisions]
    bits = [float(rows[p]["effective_bits"]) for p in precisions]
    labels = [f"{LABELS[p]}\n({bits[i]:.2f} bit)" for i, p in enumerate(precisions)]
    colors = [COLORS[p] for p in precisions]

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    bars = ax.bar(labels, accs, color=colors, width=0.62, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Best Validation Accuracy (%)", fontsize=11)
    ax.set_title("ViT Quantization Precision Comparison on flower_photos", fontsize=13, pad=12)
    ax.set_ylim(min(accs) - 4, max(accs) + 4)
    ax.axhline(accs[precisions.index("fp32")], color="#94a3b8", linestyle="--", linewidth=1, label="FP32 baseline")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.25,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print("saved", out_path)


def plot_curves(histories, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    for p in ORDER:
        if p not in histories:
            continue
        h = histories[p]["history"]
        xs = [e["epoch"] for e in h]
        ys = [e["val_acc"] for e in h]
        ax.plot(xs, ys, label=LABELS[p], color=COLORS[p], linewidth=2)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=11)
    ax.set_title("Validation Accuracy Curves (30 epochs)", fontsize=13, pad=12)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print("saved", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=Path("runs/precision_compare"))
    ap.add_argument("--out-dir", type=Path, default=Path("docs/assets"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_summary(args.run_dir)
    hist = load_histories(args.run_dir)
    plot_bar(rows, args.out_dir / "precision_compare_bar.png")
    plot_curves(hist, args.out_dir / "precision_compare_curves.png")


if __name__ == "__main__":
    main()
