#!/usr/env/bin python3.8

from pprint import pprint
from functools import reduce
from operator import mul
from typing import Dict, List, cast

import torch
import numpy as np
from torch import Tensor, einsum
import torch.nn.functional as F

from utils import simplex
from utils import soft_length


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.nd: str = kwargs["nd"]
        print(f"> Initialized {self.__class__.__name__} with kwargs:")
        pprint(kwargs)

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor, ___) -> Tensor:
        assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = cast(Tensor, target[:, self.idc, ...].type(torch.float32))

        loss = - einsum(f"bk{self.nd},bk{self.nd}->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class AbstractConstraints():
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]
        self.nd: str = kwargs["nd"]
        self.C = len(self.idc)
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])

        print(f"> Initialized {self.__class__.__name__} with kwargs:")
        pprint(kwargs)

    def penalty(self, z: Tensor) -> Tensor:
        """
        id: int - Is used to tell if is it the upper or the lower bound
                  0 for lower, 1 for upper
        """
        raise NotImplementedError

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor, filenames: List[str]) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        assert probs.shape == target.shape

        # b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        b: int
        b, _, *im_shape = probs.shape
        _, _, k, two = bounds.shape  # scalar or vector
        assert two == 2

        value: Tensor = cast(Tensor, self.__fn__(probs[:, self.idc, ...]))
        lower_b = bounds[:, self.idc, :, 0]
        upper_b = bounds[:, self.idc, :, 1]

        assert value.shape == (b, self.C, k), value.shape
        assert lower_b.shape == upper_b.shape == (b, self.C, k), lower_b.shape

        upper_z: Tensor = cast(Tensor, (value - upper_b).type(torch.float32)).reshape(b, self.C * k)
        lower_z: Tensor = cast(Tensor, (lower_b - value).type(torch.float32)).reshape(b, self.C * k)
        assert len(upper_z) == len(lower_b) == len(filenames)

        upper_penalty: Tensor = self.penalty(upper_z)
        lower_penalty: Tensor = self.penalty(lower_z)
        assert upper_penalty.numel() == lower_penalty.numel() == upper_z.numel() == lower_z.numel()

        # f for flattened axis
        res: Tensor = einsum("f->", upper_penalty) + einsum("f->", lower_penalty)

        loss: Tensor = res.sum() / reduce(mul, im_shape)
        assert loss.requires_grad == probs.requires_grad  # Handle the case for validation

        return loss


class NaivePenalty(AbstractConstraints):
    def penalty(self, z: Tensor) -> Tensor:
        # assert z.shape == ()

        return F.relu(z)**2


class LogBarrierLoss(AbstractConstraints):
    def __init__(self, **kwargs):
        self.t: float = kwargs["t"]
        super().__init__(**kwargs)

    def penalty(self, z: Tensor) -> Tensor:
        # assert z.shape == ()
        z_: Tensor = z.flatten()
        # del z

        barrier_part: Tensor = - torch.log(-z_) / self.t  # Careful, this part can produce NaN
        barrier_part[torch.isnan(barrier_part)] = 0
        linear_part: Tensor = self.t * z_ + -np.log(1 / (self.t**2)) / self.t + 1 / self.t
        assert barrier_part.dtype == linear_part.dtype == torch.float32

        below_threshold: Tensor = z_ <= - 1 / self.t**2
        assert below_threshold.dtype == torch.bool

        assert barrier_part.shape == linear_part.shape == below_threshold.shape
        res = barrier_part * below_threshold + linear_part * (~below_threshold)
        assert res.dtype == torch.float32

        # if z <= - 1 / self.t**2:
        #     res = - torch.log(-z) / self.t
        # else:
        #     res = self.t * z + -np.log(1 / (self.t**2)) / self.t + 1 / self.t

        assert res.requires_grad == z.requires_grad
        # print(res)

        return res


class LengthRatioLoss():
    def __init__(self, **kwargs):
        self.class_pair: tuple[int, int] = kwargs["class_pair"]
        self.bounds: tuple[int, int] = kwargs["bounds"]
        self.nd: str = kwargs["nd"]
        self.t: float = kwargs["t"]
        print(f"> Initialized {self.__class__.__name__} with kwargs:")
        pprint(kwargs)

    def penalty(self, z: Tensor) -> Tensor:
        # assert z.shape == ()
        z_: Tensor = z.flatten()
        # del z

        barrier_part: Tensor = - torch.log(-z_) / self.t  # Careful, this part can produce NaN
        barrier_part[torch.isnan(barrier_part)] = 0
        linear_part: Tensor = self.t * z_ + -np.log(1 / (self.t**2)) / self.t + 1 / self.t
        assert barrier_part.dtype == linear_part.dtype == torch.float32

        below_threshold: Tensor = z_ <= - 1 / self.t**2
        assert below_threshold.dtype == torch.bool

        assert barrier_part.shape == linear_part.shape == below_threshold.shape
        res = barrier_part * below_threshold + linear_part * (~below_threshold)
        assert res.dtype == torch.float32

        # if z <= - 1 / self.t**2:
        #     res = - torch.log(-z) / self.t
        # else:
        #     res = self.t * z + -np.log(1 / (self.t**2)) / self.t + 1 / self.t

        assert res.requires_grad == z.requires_grad
        # print(res)

        return res

    def __call__(self, probs: Tensor, _: Tensor, __: Tensor, ___) -> Tensor:
        assert simplex(probs)

        B, K, *_ = probs.shape  # type: ignore

        lengths: Tensor = soft_length(probs[:, self.class_pair, ...])
        assert lengths.shape == (B, 2, 1), lengths.shape

        loss: Tensor = self.penalty(self.bounds[0] - lengths[0]) + self.penalty(lengths[1] - self.bounds[1])
        assert loss.shape == (2,), loss.shape

        return loss.mean()
