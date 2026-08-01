import argparse
import os
import random
import time
import warnings

import numpy as np
import torch
from mmcv import Config
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval
import mmdet3d.runner  # noqa: F401  # register IterProgressHook / CustomEpochBasedRunner

# Keep console readable during training.
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message=".*__floordiv__.*")
warnings.filterwarnings("ignore", message=".*np\\.bool.*")
warnings.filterwarnings("ignore", message=".*np\\.long.*")


def main():
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir

    # dump full config to file only (not console)
    cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    logger.info(
        "Config file: %s | run_dir: %s | max_epochs: %s | samples_per_gpu: %s",
        args.config,
        cfg.run_dir,
        cfg.get("max_epochs", cfg.runner.get("max_epochs", "?")),
        cfg.data.get("samples_per_gpu", "?"),
    )
    logger.info("Full config saved to %s", os.path.join(cfg.run_dir, "configs.yaml"))

    if cfg.seed is not None:
        logger.info(
            "Set random seed to %s, deterministic mode: %s",
            cfg.seed,
            cfg.deterministic,
        )
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    datasets = [build_dataset(cfg.data.train)]

    model = build_model(cfg.model)
    # init_weights / Pretrained init_cfg may dump thousands of missing keys via print()
    import builtins

    _real_print = builtins.print

    def _quiet_print(*args, **kwargs):
        text = " ".join(str(a) for a in args)
        if (
            "missing keys" in text
            or "unexpected key" in text
            or "do not match exactly" in text
        ):
            return
        return _real_print(*args, **kwargs)

    builtins.print = _quiet_print
    try:
        model.init_weights()
    finally:
        builtins.print = _real_print
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %s | params: %.2fM", model.__class__.__name__, n_params / 1e6)
    train_model(
        model,
        datasets,
        cfg,
        distributed=True,
        validate=True,
        timestamp=timestamp,
    )


if __name__ == "__main__":
    main()
