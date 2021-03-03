#!/usr/bin/env python3.7

from typing import List

import torch
from torch import Tensor

from utils import eq


class ConstantBounds():
    def __init__(self, **kwargs):
        self.C: int = kwargs['C']
        self.const: Tensor = torch.zeros((self.C, 1, 2), dtype=torch.float32)

        for i, (low, high) in kwargs['values'].items():
            self.const[i, 0, 0] = low
            self.const[i, 0, 1] = high

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        return self.const


class TagBounds(ConstantBounds):
    def __init__(self, **kwargs):
        super().__init__(C=kwargs['C'], values=kwargs["values"])  # We use it as a dummy

        self.idc: List[int] = kwargs['idc']
        self.ignore_disp: bool
        if 'ignore_disp' in kwargs:
            self.ignore_disp = kwargs['ignore_disp']
        else:
            self.ignore_disp = False
        self.idc_mask: Tensor = torch.zeros(self.C, dtype=torch.uint8)  # Useful to mask the class booleans
        self.idc_mask[self.idc] = 1

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        positive_class: Tensor = torch.einsum("cwh->c", target) > 0
        weak_positive_class: Tensor = torch.einsum("cwh->c", weak_target) > 0

        masked_positive: Tensor = torch.einsum("c,c->c", positive_class, self.idc_mask).type(torch.float32)  # Keep only the idc
        masked_weak: Tensor = torch.einsum("c,c->c", weak_positive_class, self.idc_mask).type(torch.float32)
        assert eq(masked_positive, masked_weak) or self.ignore_disp, f"Unconsistent tags between labels: {filename}"

        res: Tensor = super().__call__(image, target, weak_target, filename)
        masked_res = torch.einsum("cki,c->cki", res, masked_positive)

        return masked_res


class PreciseBounds():
    def __init__(self, **kwargs):
        self.margin: float = kwargs['margin']
        self.mode: str = kwargs['mode']

        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        value: Tensor = self.__fn__(target[None, ...])[0].type(torch.float32)  # cwh and not bcwh
        margin: Tensor
        if self.mode == "percentage":
            margin = value * self.margin
        elif self.mode == "abs":
            margin = torch.ones_like(value) * self.margin
        else:
            raise ValueError("mode")

        with_margin: Tensor = torch.stack([value - margin, value + margin], dim=-1)
        assert with_margin.shape == (*value.shape, 2), with_margin.shape

        res = torch.max(with_margin, torch.zeros(*value.shape, 2)).type(torch.float32)

        return res
