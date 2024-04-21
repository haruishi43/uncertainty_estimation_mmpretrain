#!/usr/bin/env python3

from typing import Optional
from warnings import warn

from mmengine.hooks import Hook
from mmengine.model.wrappers import is_model_wrapper
from mmengine.runner import Runner

from mmpretrain.registry import HOOKS


@HOOKS.register_module()
class PassStepInfoHook(Hook):
    """Set runner's step information to the model."""

    def before_train_epoch(self, runner: Runner) -> None:
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        if hasattr(model, "set_current_step_from_epoch"):
            model.set_current_step_from_epoch(
                runner.epoch,
                runner.max_epochs,
            )
        else:
            warn(
                "The model does not have the method `set_current_step_from_epoch`. "
                "The hook will not work."
            )

    def before_train_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: Optional[dict] = None,
        outputs: Optional[dict] = None,
    ) -> None:
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        if hasattr(model, "set_current_step_from_iter"):
            model.set_current_step_from_iter(
                runner.iter,
                runner.max_iters,
            )
        else:
            warn(
                "The model does not have the method `set_current_step_from_iter`. "
                "The hook will not work."
            )
