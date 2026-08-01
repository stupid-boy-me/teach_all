from mmcv.runner import HOOKS, Hook

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


@HOOKS.register_module()
class IterProgressHook(Hook):
    """Per-iteration tqdm progress bar with loss / lr / eta."""

    def __init__(self, by_epoch=True):
        self.by_epoch = by_epoch
        self.bar = None

    def before_train_epoch(self, runner):
        if tqdm is None or not getattr(runner, "data_loader", None):
            return
        total = len(runner.data_loader)
        self.bar = tqdm(
            total=total,
            desc=f"Epoch [{runner.epoch + 1}/{runner.max_epochs}]",
            dynamic_ncols=True,
            leave=True,
            mininterval=0.5,
        )

    def after_train_iter(self, runner):
        if self.bar is None:
            return
        postfix = {}
        outputs = getattr(runner, "outputs", None) or {}
        if "loss" in outputs:
            try:
                postfix["loss"] = f"{float(outputs['loss']):.4f}"
            except Exception:
                pass
        if hasattr(runner, "current_lr"):
            try:
                lrs = runner.current_lr()
                if isinstance(lrs, (list, tuple)) and lrs:
                    postfix["lr"] = f"{float(lrs[0]):.2e}"
                elif isinstance(lrs, dict) and lrs:
                    postfix["lr"] = f"{float(next(iter(lrs.values()))[0]):.2e}"
            except Exception:
                pass
        if postfix:
            self.bar.set_postfix(postfix, refresh=False)
        self.bar.update(1)

    def after_train_epoch(self, runner):
        if self.bar is not None:
            self.bar.close()
            self.bar = None
