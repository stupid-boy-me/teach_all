#!/usr/bin/env python3
"""Train / evaluate ViT under fp32, int8, int4, int2, bit158 on flower_photos.

Example:
  python train_compare.py --data-dir /workspace/datasets/flower_photos --epochs 5 --precisions all
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from quant_linear import SUPPORTED_PRECISIONS, count_effective_bits
from vit_model import build_vit


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    max_per_class: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    full_train = datasets.ImageFolder(data_dir, transform=train_tf)
    full_eval = datasets.ImageFolder(data_dir, transform=eval_tf)
    assert full_train.classes == full_eval.classes

    # Stratified split + optional per-class cap for quick validation.
    rng = random.Random(seed)
    class_to_indices: Dict[int, List[int]] = {i: [] for i in range(len(full_train.classes))}
    for idx, (_, y) in enumerate(full_train.samples):
        class_to_indices[y].append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []
    for y, indices in class_to_indices.items():
        indices = indices[:]
        rng.shuffle(indices)
        if max_per_class > 0:
            indices = indices[:max_per_class]
        n_val = max(1, int(len(indices) * val_ratio))
        val_idx.extend(indices[:n_val])
        train_idx.extend(indices[n_val:])

    print(
        f"Dataset classes={full_train.classes} | "
        f"train={len(train_idx)} val={len(val_idx)} "
        f"(val_ratio={val_ratio}, max_per_class={max_per_class or 'all'})"
    )

    train_loader = DataLoader(
        Subset(full_train, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        Subset(full_eval, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, full_train.classes


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / max(total, 1), loss_sum / max(total, 1)


def train_one(
    precision: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    model_size: str,
    out_dir: Path,
) -> Dict:
    model = build_vit(precision=precision, num_classes=num_classes, model_size=model_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_acc = -1.0
    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
        scheduler.step()

        train_loss = running_loss / max(seen, 1)
        val_acc, val_loss = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"[{precision}] epoch {epoch}/{epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.2f}%"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), out_dir / f"vit_{precision}_best.pth")

    train_seconds = time.time() - t0
    # Reload best for final timing / accuracy.
    # weights_only=True 仅新版 PyTorch 支持；旧版（如 1.x）会报错，故兼容两种写法。
    ckpt_path = out_dir / f"vit_{precision}_best.pth"
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    infer_t0 = time.time()
    final_acc, final_loss = evaluate(model, val_loader, device)
    infer_seconds = time.time() - infer_t0

    n_params = sum(p.numel() for p in model.parameters())
    result = {
        "precision": precision,
        "effective_bits": count_effective_bits(precision),
        "num_params": n_params,
        "best_val_acc": best_acc,
        "final_val_acc": final_acc,
        "final_val_loss": final_loss,
        "train_seconds": train_seconds,
        "infer_seconds": infer_seconds,
        "epochs": epochs,
        "history": history,
    }
    with open(out_dir / f"result_{precision}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare ViT precisions on flower_photos")
    p.add_argument(
        "--data-dir",
        type=str,
        default="/workspace/datasets/flower_photos",
        help="ImageFolder root (class subdirs). Windows E:\\WSL\\... maps to /mnt/e/... if mounted.",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Cap samples per class for a quick sanity check (0 = use all).",
    )
    p.add_argument("--model-size", choices=["tiny", "small"], default="tiny")
    p.add_argument(
        "--precisions",
        type=str,
        default="all",
        help="Comma-separated list or 'all'. Choices: " + ",".join(SUPPORTED_PRECISIONS),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="runs/precision_compare")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = args.data_dir
    # Convenience: allow Windows-style path if user pasted it into WSL.
    if data_dir.startswith("E:\\") or data_dir.startswith("E:/"):
        alt = "/mnt/e/" + data_dir[3:].replace("\\", "/")
        if os.path.isdir(alt):
            data_dir = alt
        else:
            print(f"Windows path not mounted at {alt}, falling back to {args.data_dir}")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Dataset not found: {data_dir}\n"
            f"Put flower_photos (ImageFolder layout) there, or pass --data-dir."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if args.precisions.strip().lower() == "all":
        precisions = list(SUPPORTED_PRECISIONS)
    else:
        precisions = [x.strip() for x in args.precisions.split(",") if x.strip()]
        for p in precisions:
            if p not in SUPPORTED_PRECISIONS:
                raise ValueError(f"unsupported precision: {p}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, classes = build_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )

    results = []
    for precision in precisions:
        print("=" * 60)
        print(f"Training precision={precision}")
        result = train_one(
            precision=precision,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=len(classes),
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            model_size=args.model_size,
            out_dir=out_dir,
        )
        results.append(result)

    results_sorted = sorted(results, key=lambda r: r["best_val_acc"], reverse=True)
    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "precision",
                "effective_bits",
                "best_val_acc",
                "final_val_acc",
                "final_val_loss",
                "train_seconds",
                "infer_seconds",
                "num_params",
                "epochs",
            ],
        )
        writer.writeheader()
        for r in results_sorted:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    print("\n===== Precision Comparison (sorted by best_val_acc) =====")
    print(f"{'precision':<10} {'bits':>6} {'best_acc':>10} {'infer_s':>10}")
    for r in results_sorted:
        print(
            f"{r['precision']:<10} {r['effective_bits']:>6.2f} "
            f"{r['best_val_acc']:>9.2f}% {r['infer_seconds']:>10.2f}"
        )
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
