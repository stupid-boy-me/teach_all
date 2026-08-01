from mmcv.runner import EpochBasedRunner
from mmcv.runner.builder import RUNNERS


@RUNNERS.register_module()
class CustomEpochBasedRunner(EpochBasedRunner):
    def set_dataset(self, dataset):
        self._dataset = dataset

    def train(self, data_loader, **kwargs):
        # update the schedule for data augmentation
        for dataset in self._dataset:
            dataset.set_epoch(self.epoch)
        super().train(data_loader, **kwargs)

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        # Skip the long hook-order dump that clutters the console.
        self.get_hook_info = lambda: "(hook list omitted)"
        return super().run(data_loaders, workflow, max_epochs=max_epochs, **kwargs)
