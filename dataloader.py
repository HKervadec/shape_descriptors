#!/usr/env/bin python3.6

import io
import re
import random
from pathlib import Path
from pprint import pprint
from itertools import repeat
from operator import itemgetter
from functools import partial
from multiprocessing import cpu_count
from typing import Callable, BinaryIO, Match, Pattern, Tuple, Union, Optional

import torch
import numpy as np
from PIL import Image
from torch import Tensor
from torchvision import transforms
from skimage.transform import resize
from torch._six import container_abcs
from torch.utils.data import Dataset, DataLoader, Sampler

from utils import map_, class2one_hot, id_
from utils import one_hot, depth
from utils import pre_augment

F = Union[Path, BinaryIO]
D = Union[Image.Image, np.ndarray, Tensor]


resizing_fn = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)


def png_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
            lambda img: img.convert('L'),
            lambda img: np.array(img)[np.newaxis, ...],
            lambda nd: nd / 255,  # max <= 1
            lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])


def png_transform_npy(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
            lambda img: img.convert('L'),
            lambda img: np.array(img)[np.newaxis, ...],
            lambda nd: nd / 255,  # max <= 1
        ])


def npy_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
            lambda npy: np.array(npy)[np.newaxis, ...],
            lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])


def raw_npy_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
            lambda npy: np.array(npy),
            lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])


def from_numpy_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
            lambda nd: torch.tensor(nd)
        ])


def tensor_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
            lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])


def gt_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
            lambda img: np.array(img)[...],
            lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
            partial(class2one_hot, K=K),
            itemgetter(0)  # Then pop the element to go back to img shape
        ])


def dummy_gt_transform(resolution: Tuple[float, ...], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
            lambda img: np.array(img)[...],
            lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
            lambda t: torch.zeros_like(t),
            partial(class2one_hot, K=K),
            itemgetter(0)  # Then pop the element to go back to img shape
        ])


def get_loaders(args, data_folder: str,
                batch_size: int, n_class: int,
                debug: bool, in_memory: bool,
                dimensions: int,
                use_spacing: bool = False) -> Tuple[list[DataLoader], list[DataLoader]]:
        losses_list = eval(args.losses)
        if depth(losses_list) == 1:
                losses_list = [losses_list]

        list_bounds_generators: list[list[Callable]] = []
        for losses in losses_list:
                tmp = []

                for _, _, bounds_name, bounds_params, fn, _ in losses:
                        if bounds_name is None:
                                tmp.append(lambda *a: torch.zeros(n_class, 1, 2))
                                continue

                        bounds_class = getattr(__import__('bounds'), bounds_name)
                        tmp.append(bounds_class(C=args.n_class, fn=fn, **bounds_params))
                list_bounds_generators.append(tmp)

        list_folders_list = eval(args.folders)
        if depth(list_folders_list) == 1:  # For compatibility reasons, avoid changing all the previous configuration files
                list_folders_list = [list_folders_list]
        # print(folders_list)

        # Prepare the datasets and dataloaders
        print()
        train_loaders = []
        for i, (train_topfolder, folders_list, bounds_generators) in \
                enumerate(zip(args.training_folders, list_folders_list, list_bounds_generators)):

                folders, trans, are_hots = zip(*folders_list)
                print(f">> {i}th training loader: {train_topfolder} with {folders}")

                partial_aug: Optional[Callable]
                if any([args.augment_blur, args.augment_rotate, args.augment_scale]):
                        partial_aug = partial(pre_augment,
                                              blur=args.augment_blur,
                                              blur_onlyfirst=args.blur_onlyfirst,
                                              rotate=args.augment_rotate,
                                              scale=args.augment_scale)
                else:
                        partial_aug = None

                # Create partial functions: Easier for readability later (see the difference between train and validation)
                gen_dataset = partial(SliceDataset,
                                      transforms=trans,
                                      are_hots=are_hots,
                                      debug=debug,
                                      K=n_class,
                                      in_memory=in_memory,
                                      bounds_generators=bounds_generators,
                                      dimensions=dimensions,
                                      no_assert=args.no_assert_dataloader,
                                      ignore_norm=args.ignore_norm_dataloader)
                data_loader = partial(DataLoader,
                                      num_workers=min(cpu_count(), batch_size + 5),
                                      pin_memory=True,
                                      collate_fn=custom_collate)

                train_folders: list[Path] = [Path(data_folder, train_topfolder, f) for f in folders]
                # I assume all files have the same name inside their folder: makes things much easier
                train_names: list[str] = map_(lambda p: str(p.name), train_folders[0].glob("*"))
                train_set = gen_dataset(train_names,
                                        train_folders,
                                        augment=partial_aug)
                if args.group_train:
                        train_sampler = PatientSampler(train_set, args.grp_regex, shuffle=True)
                        train_loader = data_loader(train_set,
                                                   batch_sampler=train_sampler)
                else:
                        train_loader = data_loader(train_set,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   drop_last=False)

                train_loaders.append(train_loader)

                if i == args.val_loader_id or (args.val_loader_id == -1 and (i + 1) == len(args.training_folders)):
                        print(f">> Validation dataloader (id {args.val_loader_id}), {train_topfolder} {folders}")
                        val_folders: list[Path] = [Path(data_folder, args.validation_folder, f) for f in folders]
                        val_names: list[str] = map_(lambda p: str(p.name), val_folders[0].glob("*"))
                        val_set = gen_dataset(val_names,
                                              val_folders)
                        val_sampler = PatientSampler(val_set, args.grp_regex, shuffle=False) if args.group else None
                        val_batch_size = 1 if val_sampler else batch_size
                        val_loader = data_loader(val_set,
                                                 batch_sampler=val_sampler,
                                                 batch_size=val_batch_size)

        return train_loaders, [val_loader]


class SliceDataset(Dataset):
        def __init__(self, filenames: list[str], folders: list[Path], are_hots: list[bool],
                     bounds_generators: list[Callable], transforms: list[Callable], debug=False, quiet=False,
                     K=4, in_memory: bool = False,
                     extensions: list[str] = None,
                     augment: Optional[Callable] = None, ignore_norm: bool = False,
                     dimensions: int = 2, debug_size: int = 10, no_assert: bool = False) -> None:
                self.folders: list[Path] = folders
                self.transforms: list[Callable[[Tuple, int], Callable[[D], Tensor]]] = transforms
                assert len(self.transforms) == len(self.folders)

                # Deduce it from retrocompatiblity, but ideally, we want to feed only stems to the dataloader
                self.extensions: list[str]  # Include the dot: .npy, .png...
                if extensions:
                        self.extensions = extensions
                else:
                        self.extensions = [Path(filenames[0]).suffix] * len(folders)
                assert len(self.extensions) == len(self.folders)
                # getting the stem of stems won't be an issue
                self.stems: list[str] = [Path(f).stem for f in filenames]

                self.are_hots: list[bool] = are_hots
                self.debug = debug
                self.K: int = K  # Number of classes
                self.in_memory: bool = in_memory
                self.quiet: bool = quiet
                self.bounds_generators: list[Callable] = bounds_generators
                self.augment: Optional[Callable] = augment
                self.ignore_norm: bool = ignore_norm
                self.dimensions: int = dimensions
                assert len(self.bounds_generators) == (len(self.folders) - 2)

                self.no_assert: bool = no_assert

                if self.debug:
                        self.stems = self.stems[:debug_size]

                assert self.check_files()  # Make sure all file exists

                # Load things in memory if needed
                self.files: list[list[F]] = SliceDataset.load_images(self.folders, 
                                                                     self.extensions,
                                                                     self.stems,
                                                                     self.in_memory)
                assert len(self.files) == len(self.folders) == len(self.extensions)
                for files in self.files:
                        assert len(files) == len(self.stems)

                if not self.quiet:
                        print(f"> Initialized {self.__class__.__name__} with {len(self)} images")
                        d = {'debug': debug, 'folders': folders, 'are_hots': are_hots,
                             'in_memory': in_memory, 'augment': augment,
                             'extensions': self.extensions,
                             'K': self.K,
                             'dimensions': self.dimensions,
                             'transforms': [tr.__name__ for tr in self.transforms],
                             'bounds_generators': [gen for gen in self.bounds_generators]}  # type: ignore
                        pprint(d)

        def check_files(self) -> bool:
                print(self.extensions)
                print(self.folders)
                folder: Path
                for folder, ext in zip(self.folders, self.extensions):
                        if not Path(folder).exists():
                                return False

                        for stem in self.stems:
                                if not (folder / stem).with_suffix(ext).exists():
                                        return False

                return True

        @staticmethod
        def load_images(folders: list[Path], extensions: list[str],
                        stems: list[str], in_memory: bool, quiet=False) -> list[list[F]]:
                def load(folder: Path, stem: str, ext: str) -> F:
                        p: Path = (folder / stem).with_suffix(ext)
                        if in_memory:
                                with open(p, 'rb') as data:
                                        res = io.BytesIO(data.read())
                                return res
                        return p
                if in_memory and not quiet:
                        print("> Loading the data in memory...")

                files: list[list[F]] = [[load(f, stem, ext)
                                         for stem in stems]
                                        for f, ext in zip(folders, extensions)]

                return files

        def __len__(self):
                return len(self.stems)

        def __load__(self, files: list[F], index: int, ext: str):
                if ext == ".png":
                        return Image.open(files[index])
                elif ext == ".npy":
                        return np.load(files[index])

                raise ValueError(ext)

        def __getitem__(self, index: int) -> dict[str, Union[str,
                                                             int,
                                                             Tensor,
                                                             list[Tensor],
                                                             list[Tuple[slice, ...]],
                                                             list[Tuple[Tensor, Tensor]]]]:
                stem: str = self.stems[index]
                images: list[D] = [self.__load__(files, index, ext)
                                   for (files, ext) in zip(self.files, self.extensions)]

                # Pre-augment (if needed) -- to be done BEFORE the other transforms:
                if self.augment:
                        images = self.augment(images)

                resolution: Tuple[float, ...]
                resolution = tuple([1] * self.dimensions)

                # Final transforms and assertions
                assert len(images) == len(self.folders) == len(self.transforms)
                final_tensors: list[Tensor] = [tr(resolution, self.K)(e) for (tr, e) in zip(self.transforms, images)]
                _, *img_shape = final_tensors[0].shape

                if not self.no_assert:
                        # main image is between 0 and 1
                        if not self.ignore_norm:
                                assert 0 <= final_tensors[0].min() and final_tensors[0].max() <= 1, \
                                    (final_tensors[0].min(), final_tensors[0].max())

                        for ttensor in final_tensors[1:]:  # Things should be one-hot or at least have the shape
                                assert ttensor.shape == (self.K, *img_shape), (ttensor.shape, self.K, *img_shape)

                        for ttensor, is_hot in zip(final_tensors, self.are_hots):  # All masks (ground truths) are class encoded
                                if is_hot:
                                        assert one_hot(ttensor, axis=0), torch.einsum("k...->...", ttensor)

                img, gt = final_tensors[:2]

                bounds: list[Tensor]
                bounds = [f(img, gt, t, stem) for f, t in zip(self.bounds_generators, final_tensors[2:])]

                patches: list[Tuple[slice, ...]]
                if self.dimensions == 2:  # Everything fits within one patch
                        patches = [tuple([slice(0, e) for e in img_shape])]  # list of patches, slice for each dimension
                elif self.dimensions >= 2:
                        raise NotImplementedError("REDACTED")

                return {'filenames': stem,
                        'images': final_tensors[0],
                        'gt': final_tensors[1],
                        'labels': final_tensors[2:],
                        'bounds': bounds,
                        'samplings': patches,
                        'index': index}


_use_shared_memory = True


def custom_collate(batch):
        """Collate function to handle dict from dataset dict[str, Union[str, Tensor, list[Tensor], list[slice]]]"""

        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.sparse.FloatTensor):
                return batch
        elif isinstance(elem, torch.Tensor):
                out = None
                # if torch.utils.data.get_worker_info() is not None:
                if _use_shared_memory:
                        # If we're in a background process, concatenate directly into a
                        # shared memory tensor to avoid an extra copy
                        numel = sum([x.numel() for x in batch])
                        storage = elem.storage()._new_shared(numel)
                        out = elem.new(storage)
                return torch.stack(batch, 0, out=out)
        elif isinstance(elem, np.ndarray):
                return np.stack(batch)
        elif isinstance(elem, str) or isinstance(elem, slice) or isinstance(elem, int):
                return batch
        elif isinstance(elem, container_abcs.Mapping):
                return {key: custom_collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, list):
                if len(elem) == 0:
                        return batch

                if isinstance(elem[0], tuple):  # Handling for spacings
                        return batch

                transposed = zip(*batch)
                return [custom_collate(samples) for samples in transposed]

        raise TypeError(elem_type)


class PatientSampler(Sampler):
        def __init__(self, dataset: SliceDataset, grp_regex, shuffle=False, quiet=False) -> None:
                stems: list[str] = dataset.stems
                # Might be needed in case of escape sequence fuckups
                # self.grp_regex = bytes(grp_regex, "utf-8").decode('unicode_escape')
                assert grp_regex is not None
                self.grp_regex = grp_regex

                # Configure the shuffling function
                self.shuffle: bool = shuffle
                self.shuffle_fn: Callable = (lambda x: random.sample(x, len(x))) if self.shuffle else id_

                # print(f"Grouping using {self.grp_regex} regex")
                # assert grp_regex == "(patient\d+_\d+)_\d+"
                # grouping_regex: Pattern = re.compile("grp_regex")
                grouping_regex: Pattern = re.compile(self.grp_regex)

                # stems: list[str] = [Path(filename).stem for filename in filenames]  # avoid matching the extension
                matches: list[Match] = map_(grouping_regex.match, stems)
                patients: list[str] = [match.group(1) for match in matches]

                unique_patients: list[str] = list(set(patients))
                assert len(unique_patients) < len(stems)
                if not quiet:
                        print(f"> Found {len(unique_patients)} unique patients out of {len(stems)} images ; regex: {self.grp_regex}")

                self.idx_map: dict[str, list[int]] = dict(zip(unique_patients, repeat(None)))
                for i, patient in enumerate(patients):
                        if not self.idx_map[patient]:
                                self.idx_map[patient] = []

                        self.idx_map[patient] += [i]
                # print(self.idx_map)
                assert sum(len(self.idx_map[k]) for k in unique_patients) == len(stems)

                for pid in self.idx_map.keys():
                        self.idx_map[pid] = sorted(self.idx_map[pid], key=lambda i: stems[i])

                # print("Patient to slices mapping done")

        def __len__(self):
                return len(self.idx_map.keys())

        def __iter__(self):
                values = list(self.idx_map.values())
                shuffled = self.shuffle_fn(values)
                return iter(shuffled)
