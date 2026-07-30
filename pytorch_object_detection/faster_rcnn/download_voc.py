"""Download PASCAL VOC 2007 (trainval + test)."""
from __future__ import annotations

import argparse
from pathlib import Path

from torchvision.datasets import VOCDetection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="./data",
        help="下载根目录；完成后会有 <root>/VOCdevkit/VOC2007/",
    )
    parser.add_argument("--year", type=str, default="2007")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading VOC{args.year} into {root.resolve()} ...")
    for split in ("trainval", "test"):
        print(f"  -> {split}")
        VOCDetection(root=str(root), year=args.year, image_set=split, download=True)
    print("Done.")
    print(f"请保持 configs/default.yaml 中 data.root = {root.as_posix()}")


if __name__ == "__main__":
    main()
