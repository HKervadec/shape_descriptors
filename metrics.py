#!/usr/bin/env python3.9

import argparse
from pathlib import Path

import torch
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader

from dataloader import SliceDataset, PatientSampler, png_transform, gt_transform, custom_collate
from utils import dice_batch, dice_coef
from utils import map_, tqdm_, one_hot, simplex


def runInference(args: argparse.Namespace, pred_folder: str, pred_transform: str):
        # print('>>> Loading the data')
        device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
        K: int = args.num_classes
        n_epoch: int = 1  # To reuse same scripts between train, val and test

        folders: list[Path] = [Path(args.gt_folder),
                               Path(args.gt_folder),
                               Path(pred_folder)]  # First one is dummy
        extensions: list[str] = [args.extensions[0]] + args.extensions  # Since we double the first folder
        stems: list[str] = map_(lambda p: str(p.stem), folders[0].glob(f"*{extensions[0]}"))

        pred_tr_fn = getattr(__import__("dataloader"), pred_transform)
        dt_set = SliceDataset(stems,
                              folders,
                              extensions=extensions,
                              transforms=[png_transform, gt_transform, pred_tr_fn],
                              are_hots=[False, True, False],
                              bounds_generators=[lambda *a: torch.zeros(K, 1, 2)],
                              debug=args.debug,
                              K=K)
        sampler = PatientSampler(dt_set, args.grp_regex)
        loader = DataLoader(dt_set,
                            batch_sampler=sampler,
                            num_workers=11,
                            shuffle=False,
                            drop_last=False,
                            collate_fn=custom_collate)

        # print('>>> Computing the metrics')
        total_iteration, total_images = len(loader), len(loader.dataset)
        metrics: dict[str, Tensor] = {}
        if "dice" in args.metrics:
                metrics["dice"] = torch.zeros((n_epoch, total_images, K), dtype=torch.float32)
        if "3d_dsc" in args.metrics:
                metrics["3d_dsc"] = torch.zeros((n_epoch, total_iteration, K), dtype=torch.float32)

        desc = ">> Computing"
        tq_iter = tqdm_(enumerate(loader), total=total_iteration, desc=desc)
        done: int = 0
        with torch.no_grad():
                for j, data in tq_iter:
                        gt: Tensor = data["gt"].to(device)
                        pred: Tensor = data["labels"][0].to(device)

                        assert pred.shape == gt.shape
                        assert simplex(pred)  # Predictions could be one-hot or probabilities
                        assert one_hot(gt), (gt.shape)

                        B, K_, W, H = gt.shape
                        assert K == K_

                        if "dice" in metrics.keys():
                                dices: Tensor = dice_coef(pred, gt)
                                assert dices.shape == (B, K), (dices.shape, B, K)

                                metrics["dice"][0, done:done + B, ...] = dices
                        if "3d_dsc" in metrics.keys():
                                three_d_DSC: Tensor = dice_batch(pred, gt)
                                assert three_d_DSC.shape == (K,)

                                metrics["3d_dsc"][0, j, ...] = three_d_DSC

                        done += B

        print(f">>> {pred_folder}")
        for key, v in metrics.items():
                print(key, v[0].mean(dim=0).cpu().numpy())

        if args.save_folder:
                savedir: Path = Path(args.save_folder)
                for k, e in metrics.items():
                        dest: Path = savedir / f"{args.mode}{'_' if args.mode else ''}{k}.npy"
                        assert not dest.exists() or args.overwrite

                        np.save(dest, e.cpu().numpy())


def get_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description='Compute metrics for a list of images')
        parser.add_argument('--pred_folders', type=str, nargs='+',
                            help="The folder containing the predictions")
        parser.add_argument('--pred_transforms', type=str, nargs='+',
                            help="How to transform the predictions")
        parser.add_argument('--extensions', type=str, nargs='+',
                            choices=[".npy", ".png"])
        parser.add_argument('--gt_folder', type=str, required=True)
        parser.add_argument("--grp_regex", type=str, required=True)
        parser.add_argument("--debug", action="store_true", help="Dummy for compatibility")
        parser.add_argument('--num_classes', type=int, required=True)
        parser.add_argument('--metrics', type=str, nargs='+', 
                            choices=["dice", "3d_dsc"])

        parser.add_argument("--cpu", action="store_true")
        parser.add_argument("--overwrite", action="store_true",
                            help="Overwrite existing metrics output, without prompt.")

        parser.add_argument('--save_folder', type=str, default="", help="The folder to save the metrics")
        parser.add_argument('--mode', type=str, default="")

        args = parser.parse_args()
        assert len(args.pred_folders) == len(args.pred_transforms)
        assert len(args.extensions) == 2

        print(args)

        return args


def main() -> None:
        args = get_args()
        for (pred_folder, pred_transforms) in zip(args.pred_folders, args.pred_transforms):
                runInference(args, pred_folder, pred_transforms)


if __name__ == '__main__':
        main()
