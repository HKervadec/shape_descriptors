#!/usr/bin/env python3.9

import argparse
from pathlib import Path
from itertools import starmap
from multiprocessing.pool import Pool
from random import random, uniform, randint
from functools import lru_cache, partial, reduce

from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union, cast

import torch
import numpy as np
import scipy as sp
import torch.sparse
import torchvision.transforms.functional as tvF
from tqdm import tqdm
from torch import einsum
from torch import Tensor
from skimage.io import imsave
from PIL import Image, ImageOps


colors = ["c", "r", "g", "b", "m", 'y', 'k', 'chartreuse', 'coral', 'gold', 'lavender',
          'silver', 'tan', 'teal', 'wheat', 'orchid', 'orange', 'tomato']

# functions redefinitions
tqdm_ = partial(tqdm, dynamic_ncols=True,
                leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [' '{rate_fmt}{postfix}]')

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", Tensor, np.ndarray)


def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
                return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
                return False
        else:
                raise argparse.ArgumentTypeError('Boolean value expected.')


def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
        return list(map(fn, iter))


def starmap_(fn: Callable[[Tuple[A]], B], iter: Iterable[Tuple[A]]) -> List[B]:
        return list(starmap(fn, iter))


def mmap_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
        return Pool().map(fn, iter)


def starmmap_(fn: Callable[[Tuple[A]], B], iter: Iterable[Tuple[A]]) -> List[B]:
        return Pool().starmap(fn, iter)


def uc_(fn: Callable) -> Callable:
        return partial(uncurry, fn)


def uncurry(fn: Callable, args: List[Any]) -> Any:
        return fn(*args)


def id_(x):
        return x


def flatten_(to_flat: Iterable[Iterable[A]]) -> List[A]:
        return [e for l in to_flat for e in l]


def flatten__(to_flat):
        if type(to_flat) != list:
                return [to_flat]

        return [e for l in to_flat for e in flatten__(l)]


def depth(e: List) -> int:
        """
        Compute the depth of nested lists
        """
        if type(e) == list and e:
                return 1 + depth(e[0])

        return 0


def compose(fns, init):
        return reduce(lambda acc, f: f(acc), fns, init)


def compose_acc(fns, init):
        return reduce(lambda acc, f: acc + [f(acc[-1])], fns, [init])


# f . g (x) := f(g(x))
def c(*fns: Callable[[Any], Any]) -> Callable[[Any], Any]:
        return reduce(lambda acc, fn: (lambda x: acc(fn(x))), fns, lambda x: x)


# fns
def soft_size(a: Tensor) -> Tensor:
        return torch.einsum("bk...->bk", a)[..., None]


def batch_soft_size(a: Tensor) -> Tensor:
        return torch.einsum("bk...->k", a)[..., None]


def soft_centroid(a: Tensor) -> Tensor:
        b, k, *img_shape = a.shape
        nd: str = "whd" if len(img_shape) == 3 else "wh"

        grids = np.mgrid[[slice(0, l) for l in img_shape]]
        tensor_grids = map_(lambda e: torch.tensor(e).to(a.device).type(torch.float32), grids)

        # Make sure all grids have the same shape as img_shape
        all_shapes: list[list[int]] = [*map_(lambda e: list(e.shape), tensor_grids), img_shape]
        assert len(set(map_(lambda e: tuple(e), all_shapes))) == 1

        flotted = a.type(torch.float32)
        tot = einsum("bk...->bk", flotted) + 1e-10
        assert tot.dtype == torch.float32

        centroids = [einsum(f"bk{nd},{nd}->bk", flotted, grid) / tot for grid in tensor_grids]
        assert all(e.dtype == torch.float32 for e in centroids), map_(lambda e: e.dtype, centroids)

        res = torch.stack(centroids, dim=2)
        assert res.shape == (b, k, len(img_shape))
        assert res.dtype == torch.float32

        return res


def soft_dist_centroid(a: Tensor) -> Tensor:
        b, k, *img_shape = a.shape
        if len(img_shape) > 2:
                raise NotImplementedError("Only handle 2D for now, require to update the einsums")
        # nd: str = "whd" if len(img_shape) == 3 else "wh"

        grids: list[np.ndarray] = np.mgrid[[slice(0, l) for l in img_shape]]
        tensor_grids: list[Tensor] = map_(lambda e: torch.tensor(e).to(a.device).type(torch.float32), grids)

        # Make sure all grids have the same shape as img_shape
        assert all(g.shape == tuple(img_shape) for g in tensor_grids)

        # Useful when `a` is a label tensor (int64)
        flotted: Tensor = a.type(torch.float32)
        sizes: Tensor = einsum("bk...->bk", flotted)
        assert sizes.dtype == torch.float32

        centroids: list[Tensor] = [einsum("bkhw,hw->bk", flotted, grid) / (sizes + 1e-10)
                                   for grid in tensor_grids]
        assert all(c.dtype == torch.float32 for c in centroids), [c.dtype for c in centroids]

        assert all(c.shape == (b, k) for c in centroids)
        assert all(g.shape == tuple(img_shape) for g in tensor_grids)
        assert len(tensor_grids) == len(centroids)
        # Now the tricky part: different centroid for each batch and class:
        # g.shape: (hw), c.shape: (bk) -> d.shape: (bkhw)
        diffs: list[Tensor] = [(g.repeat(b, k, 1, 1) - c[:, :, None, None].repeat(1, 1, *img_shape))
                               for (g, c) in zip(tensor_grids, centroids)]
        assert len(diffs) == len(img_shape)
        assert all(d.shape == (a.shape) == (b, k, *img_shape) for d in diffs)
        assert all(d.dtype == torch.float32 for d in diffs), [d.dtype for d in diffs]

        dist_centroid: list[Tensor] = [einsum("bkhw,bkhw->bk", flotted, d**2) / (sizes + 1e-10)
                                       for d in diffs]

        # pprint(dist_centroid)

        res = torch.stack([dc.sqrt() for dc in dist_centroid], dim=2)
        assert res.shape == (b, k, len(img_shape))
        assert res.dtype == torch.float32

        return res


def soft_length(a: Tensor, kernel: Tuple = None) -> Tensor:
        B, K, *img_shape = a.shape

        laplacian: Tensor = static_laplacian(*img_shape, device=a.device, kernel=kernel)
        assert laplacian.dtype == torch.float64
        N, M = laplacian.shape
        assert N == M

        results: Tensor = torch.ones((B, K, 1), dtype=torch.float32, device=a.device)
        for b in range(B):
                for k in range(K):
                        flat_slice: Tensor = a[b, k].flatten()[:, None].type(torch.float64)

                        assert flat_slice.shape == (N, 1)
                        slice_length: Tensor = flat_slice.t().mm(laplacian.mm(flat_slice))

                        assert slice_length.shape == (1, 1)
                        results[b, k, :] = slice_length[...]

        return results


def soft_compactness(a: Tensor, kernel: Tuple = None) -> Tensor:
        L: Tensor = soft_length(a, kernel)
        S: Tensor = cast(Tensor, soft_size(a).type(torch.float32))

        if (S == 0).any():
                print(f"{S=}")
                print(f"{L=}")

        assert L.shape == S.shape  # Don't want any weird broadcasting issues

        return L**2 / (S + 1e-10)


# Assert utils
def uniq(a: Tensor) -> Set:
        return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
        return uniq(a).issubset(sub)


def eq(a: Tensor, b) -> bool:
        return torch.eq(a, b).all()


def simplex(t: Tensor, axis=1) -> bool:
        _sum = cast(Tensor, t.sum(axis).type(torch.float32))
        _ones = torch.ones_like(_sum, dtype=torch.float32)
        return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
        return simplex(t, axis) and sset(t, [0, 1])


# # Metrics and shitz
def meta_dice(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> Tensor:
        assert label.shape == pred.shape
        assert one_hot(label)
        assert one_hot(pred)

        inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
        sum_sizes: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(torch.float32)

        dices: Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)

        return dices


dice_coef = partial(meta_dice, "bk...->bk")
dice_batch = partial(meta_dice, "bk...->k")  # used for 3d dice


def intersection(a: Tensor, b: Tensor) -> Tensor:
        assert a.shape == b.shape
        assert sset(a, [0, 1])
        assert sset(b, [0, 1])

        res = a & b
        assert sset(res, [0, 1])

        return res


def union(a: Tensor, b: Tensor) -> Tensor:
        assert a.shape == b.shape
        assert sset(a, [0, 1])
        assert sset(b, [0, 1])

        res = a | b
        assert sset(res, [0, 1])

        return res


def inter_sum(a: Tensor, b: Tensor) -> Tensor:
        return einsum("bk...->bk", intersection(a, b).type(torch.float32))


def union_sum(a: Tensor, b: Tensor) -> Tensor:
        return einsum("bk...->bk", union(a, b).type(torch.float32))


# switch between representations
def probs2class(probs: Tensor) -> Tensor:
        b, _, *img_shape = probs.shape
        assert simplex(probs)

        res = probs.argmax(dim=1)
        assert res.shape == (b, *img_shape)

        return res


def class2one_hot(seg: Tensor, K: int) -> Tensor:
        # Breaking change but otherwise can't deal with both 2d and 3d
        # if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
        #     return class2one_hot(seg.unsqueeze(dim=0), K)[0]

        assert sset(seg, list(range(K))), (uniq(seg), K)

        b, *img_shape = seg.shape  # type: Tuple[int, ...]

        device = seg.device
        res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device).scatter_(1, seg[:, None, ...], 1)

        assert res.shape == (b, K, *img_shape)
        assert one_hot(res)

        return res


def np_class2one_hot(seg: np.ndarray, K: int) -> np.ndarray:
        return class2one_hot(torch.from_numpy(seg.copy()).type(torch.int64), K).numpy()


def probs2one_hot(probs: Tensor) -> Tensor:
        _, K, *_ = probs.shape
        assert simplex(probs)

        res = class2one_hot(probs2class(probs), K)
        assert res.shape == probs.shape
        assert one_hot(res)

        return res


# Misc utils
def save_images(segs: Tensor, names: Iterable[str], root: Path) -> None:
        for seg, name in zip(segs, names):
                save_path = (root / name).with_suffix(".png")
                save_path.parent.mkdir(parents=True, exist_ok=True)

                if len(seg.shape) == 2:
                        imsave(str(save_path), seg.detach().cpu().numpy().astype(np.uint8))
                elif len(seg.shape) == 3:
                        np.save(str(save_path), seg.detach().cpu().numpy())
                else:
                        raise ValueError("How did you get here")


def augment(*arrs: Union[np.ndarray, Image.Image], rotate_angle: float = 45,
            flip: bool = True, mirror: bool = True,
            rotate: bool = True, scale: bool = False) -> List[Image.Image]:
        imgs: List[Image.Image] = map_(Image.fromarray, arrs) if isinstance(arrs[0], np.ndarray) else list(arrs)

        if flip and random() > 0.5:
                imgs = map_(ImageOps.flip, imgs)
        if mirror and random() > 0.5:
                imgs = map_(ImageOps.mirror, imgs)
        if rotate and random() > 0.5:
                angle: float = uniform(-rotate_angle, rotate_angle)
                imgs = map_(lambda e: e.rotate(angle), imgs)
        if scale and random() > 0.5:
                scale_factor: float = uniform(1, 1.2)
                w, h = imgs[0].size  # Tuple[int, int]
                nw, nh = int(w * scale_factor), int(h * scale_factor)  # Tuple[int, int]

                # Resize
                imgs = map_(lambda i: i.resize((nw, nh)), imgs)

                # Now need to crop to original size
                bw, bh = randint(0, nw - w), randint(0, nh - h)  # Tuple[int, int]

                imgs = map_(lambda i: i.crop((bw, bh, bw + w, bh + h)), imgs)
                assert all(i.size == (w, h) for i in imgs)

        return imgs


def pre_augment(inputs: list[Union[Tensor,
                                   Image.Image,
                                   np.ndarray]],
                rotate_angle: float = 30,
                blur: bool = False, blur_onlyfirst: bool = False,
                rotate: bool = False,
                scale: bool = False) -> list[Union[Tensor,
                                             Image.Image,
                                             np.ndarray]]:
        """
        This one is supposed to be run BEFORE the class2one_hot transform.

        Otherwise, it will still work, but *break* the one-hot encoding.
        """

        is_numpy: bool = isinstance(inputs[0], np.ndarray)
        is_pil: bool = isinstance(inputs[0], Image.Image)
        is_tensor: bool = isinstance(inputs[0], Tensor)

        results: list[Union[Tensor, Image.Images]] = map_(Image.fromarray,
                                                          inputs) if is_numpy else inputs

        if blur and random() > 0.5:
                sigma: float = uniform(0.5, 2)
                partial_blur: Callable = partial(tvF.gaussian_blur,
                                                 kernel_size=[5, 5],
                                                 sigma=[sigma, sigma])
                if blur_onlyfirst:
                        results[0] = partial_blur(results[0])
                else:
                        results = map_(partial_blur,
                                       results)
        if rotate and random() > 0.5:
                angle: float = uniform(-rotate_angle, rotate_angle)
                results = map_(partial(tvF.rotate,
                                       angle=angle,
                                       resample=False,
                                       expand=False,
                                       center=None,
                                       fill=None),
                               results)
        if scale and random() > 0.5:
                scale_factor: float = uniform(0.8, 1.2)
                h: int
                w: int
                if is_pil:
                        h, w = np.asarray(results[0]).shape
                else:
                        _, _, h, w = results[0].shape
                nh, nw = int(h * scale_factor), int(w * scale_factor)  # tuple[int, int]

                if scale_factor > 1:
                        # resize (bigger)
                        results = map_(partial(tvF.resize,
                                               size=(nh, nw),
                                               interpolation=Image.NEAREST),  # To keep the ground truth correct
                                       results)
                        # crop
                        bh, bw = randint(0, nh - h), randint(0, nw - w)  # tuple[int, int]
                        results = map_(partial(tvF.crop,
                                               top=bh, left=bw,
                                               height=h, width=w),
                                       results)
                elif scale_factor < 1:
                        # resize (smaller)
                        results = map_(partial(tvF.resize,
                                               size=(nh, nw),
                                               interpolation=Image.NEAREST),  # To keep the ground truth correct
                                       results)
                        # pad with 0s
                        Δh, Δw = h - nh, w - nw  # tuple[int, int]
                        dh, dw = randint(0, Δh), randint(0, Δw)
                        results = map_(partial(tvF.pad,
                                               # If a tuple of length 4 is provided this is the padding for the left, top, right and bottom borders respectively.
                                               padding=[dw, dh, Δw - dw, Δh - dh],
                                               padding_mode='constant', fill=0),
                                       results)
                else:  # scale_factor == 1
                        print(f"> Warning: no scaling was performed ({scale_factor=})")

        if is_numpy:
                results = [np.asarray(r) for r in results]
                assert all(isinstance(r, np.ndarray) for r in results)
        elif is_pil:
                assert all(isinstance(r, Image.Image) for r in results)
        elif is_tensor:
                assert all(isinstance(r, Tensor) for r in results)

        return results


@lru_cache()
def static_laplacian(width: int, height: int,
                     kernel: Tuple = None,
                     device=None) -> Tensor:
        """
        This function compute the weights of the graph representing img.
        The weights 0 <= w_i <= 1 will be determined from the difference between the nodes: 1 for identical value,
        0 for completely different.
        :param img: The image, as a (n,n) matrix.
        :param kernel: A binary mask of (k,k) shape.
        :param sigma: Parameter for the weird exponential at the end.
        :param eps: Other parameter for the weird exponential at the end.
        :return: A float valued (n^2,n^2) symmetric matrix. Diagonal is empty
        """
        kernel_: np.ndarray
        if kernel is None:
                kernelSize = 3

                kernel_ = np.ones((kernelSize,) * 2)
                kernel_[(kernelSize // 2,) * 2] = 0

        else:
                kernel_ = np.asarray(kernel)
        # print(kernel_)

        img_shape = (width, height)
        N = width * height

        KW, KH = kernel_.shape
        K = int(np.sum(kernel_))  # 0 or 1

        A = np.pad(np.arange(N).reshape(img_shape),
                   ((KW // 2, KW // 2), (KH // 2, KH // 2)),
                   'constant',
                   constant_values=-1)
        neighs = np.zeros((K, N), np.int64)

        k = 0
        for i in range(KW):
                for j in range(KH):
                        if kernel_[i, j] == 0:
                                continue

                        T = A[i:i + width, j:j + height]
                        neighs[k, :] = T.ravel()
                        k += 1

        T1 = np.tile(np.arange(N), K)
        T2 = neighs.flatten()
        Z = T1 <= T2
        T1, T2 = T1[Z], T2[Z]

        diff = np.ones(len(T1))
        M = sp.sparse.csc_matrix((diff, (T1, T2)), shape=(N, N))
        adj = M + M.T
        laplacian = sp.sparse.spdiags(adj.sum(0), 0, N, N) - adj
        coo_laplacian = laplacian.tocoo()

        indices: Tensor = torch.stack([torch.from_numpy(coo_laplacian.row), torch.from_numpy(coo_laplacian.col)])
        torch_laplacian = torch.sparse.FloatTensor(indices.type(torch.int64),
                                                   torch.from_numpy(coo_laplacian.data),
                                                   torch.Size([N, N])).to(device)
        assert torch_laplacian.device == device

        return torch_laplacian



@lru_cache
def adj_Ts(width: int, height: int,
           kernel: Tuple = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function compute the weights of the graph representing img.
        The weights 0 <= w_i <= 1 will be determined from the difference between the nodes: 1 for identical value,
        0 for completely different.
        :param img: The image, as a (n,n) matrix.
        :param kernel: A binary mask of (k,k) shape.
        :param sigma: Parameter for the weird exponential at the end.
        :param eps: Other parameter for the weird exponential at the end.
        :return: A float valued (n^2,n^2) symmetric matrix. Diagonal is empty
        """
        kernel_: np.ndarray
        if kernel is None:
                kernelSize = 3

                kernel_ = np.ones((kernelSize,) * 2)
                kernel_[(kernelSize // 2,) * 2] = 0

        else:
                kernel_ = np.asarray(kernel)
        # print(kernel_)

        img_shape = (width, height)
        N = width * height

        KW, KH = kernel_.shape
        K = int(np.sum(kernel_))  # 0 or 1

        A = np.pad(np.arange(N).reshape(img_shape),
                   ((KW // 2, KW // 2), (KH // 2, KH // 2)),
                   'constant',
                   constant_values=-1)
        neighs = np.zeros((K, N), np.int64)

        k = 0
        for i in range(KW):
                for j in range(KH):
                        if kernel_[i, j] == 0:
                                continue

                        T = A[i:i + width, j:j + height]
                        neighs[k, :] = T.ravel()
                        k += 1

        T1 = np.tile(np.arange(N), K)
        T2 = neighs.flatten()
        Z = T1 <= T2
        T1, T2 = T1[Z], T2[Z]

        return T1, T2
