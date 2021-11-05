"""
@Time   :   2021-01-21 10:52:47
@File   :   lr_scheduler.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import math
import warnings
from bisect import bisect_right
from typing import List

import torch
from torch.optim.lr_scheduler import _LRScheduler

__all__ = ["WarmupMultiStepLR", "WarmupCosineAnnealingLR"]


class WarmupMultiStepLR(_LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            milestones: List[int],
            gamma: float = 0.1,
            warmup_factor: float = 0.001,
            warmup_epochs: int = 2,
            warmup_method: str = "linear",
            last_epoch: int = -1,
            **kwargs,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}", milestones
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_epochs, self.warmup_factor
        )
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()


class WarmupExponentialLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(self, optimizer, gamma, last_epoch=-1, warmup_epochs=2, warmup_factor=1.0 / 3, verbose=False,
                 **kwargs):
        self.gamma = gamma
        self.warmup_method = 'linear'
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_epochs, self.warmup_factor
        )

        if self.last_epoch <= self.warmup_epochs:
            return [base_lr * warmup_factor
                    for base_lr in self.base_lrs]
        return [group['lr'] * self.gamma
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** self.last_epoch
                for base_lr in self.base_lrs]


class WarmupCosineAnnealingLR(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))
    When last_epoch=-1, sets initial lr as lr.
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            max_iters: int,
            delay_iters: int = 0,
            eta_min_lr: int = 0,
            warmup_factor: float = 0.001,
            warmup_epochs: int = 2,
            warmup_method: str = "linear",
            last_epoch=-1,
            **kwargs
    ):
        self.max_iters = max_iters
        self.delay_iters = delay_iters
        self.eta_min_lr = eta_min_lr
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        assert self.delay_iters >= self.warmup_epochs, "Scheduler delay iters must be larger than warmup iters"
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch <= self.warmup_epochs:
            warmup_factor = _get_warmup_factor_at_iter(
                self.warmup_method, self.last_epoch, self.warmup_epochs, self.warmup_factor,
            )
            return [
                base_lr * warmup_factor for base_lr in self.base_lrs
            ]
        elif self.last_epoch <= self.delay_iters:
            return self.base_lrs

        else:
            return [
                self.eta_min_lr + (base_lr - self.eta_min_lr) *
                (1 + math.cos(
                    math.pi * (self.last_epoch - self.delay_iters) / (self.max_iters - self.delay_iters))) / 2
                for base_lr in self.base_lrs]


def _get_warmup_factor_at_iter(
        method: str, iter: int, warmup_iters: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See https://arxiv.org/abs/1706.02677 for more details.
    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).
    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))
