#!/usr/bin/env python3.9

import argparse
import warnings
from pathlib import Path
from pprint import pprint
from functools import reduce
from shutil import copytree, rmtree
from operator import add, itemgetter
from typing import Any, Callable, Tuple
# from time import time

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from dataloader import get_loaders
from utils import map_
from utils import depth
from utils import probs2one_hot, probs2class
from utils import dice_coef, save_images, tqdm_, dice_batch


def setup(args, n_class: int) -> Tuple[Any, Any, Any, list[list[Callable]], list[list[float]], Callable]:
        print("\n>>> Setting up")
        cpu: bool = args.cpu or not torch.cuda.is_available()
        device = torch.device("cpu") if cpu else torch.device("cuda")

        if args.weights:
                if cpu:
                        net = torch.load(args.weights, map_location='cpu')
                else:
                        net = torch.load(args.weights)
                print(f">> Restored weights from {args.weights} successfully.")
        else:
                net_class = getattr(__import__('networks'), args.network)
                net = net_class(args.modalities, n_class).to(device)
                net.init_weights()
        net.to(device)

        optimizer: Any  # disable an error for the optmizer (ADAM and SGD not same type)
        if args.use_sgd:
                optimizer = torch.optim.SGD(net.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)
        else:
                optimizer = torch.optim.Adam(net.parameters(), lr=args.l_rate, betas=(0.9, 0.99), amsgrad=False)

        # print(args.losses)
        list_losses = eval(args.losses)
        if depth(list_losses) == 1:  # For compatibility reasons, avoid changing all the previous configuration files
                list_losses = [list_losses]

        nd: str = "whd" if args.three_d else "wh"

        loss_fns: list[list[Callable]] = []
        for i, losses in enumerate(list_losses):
                print(f">> {i}th list of losses:")
                pprint(losses)
                tmp: list[Callable] = []
                for loss_name, loss_params, _, _, fn, _ in losses:
                        loss_class = getattr(__import__('losses'), loss_name)
                        tmp.append(loss_class(**loss_params, fn=fn, nd=nd))
                loss_fns.append(tmp)

        loss_weights: list[list[float]] = [map_(itemgetter(5), losses) for losses in list_losses]

        scheduler = getattr(__import__('scheduler'), args.scheduler)(**eval(args.scheduler_params))

        return net, optimizer, device, loss_fns, loss_weights, scheduler


def do_epoch(mode: str, net: Any, device: Any, loaders: list[DataLoader], epc: int,
             list_loss_fns: list[list[Callable]], list_loss_weights: list[list[float]], K: int,
             savedir: Path = None, optimizer: Any = None,
             metric_axis: list[int] = [1], requested_metrics: list[str] = None,
             temperature: float = 1) -> dict[str, Tensor]:
        assert mode in ["train", "val", "dual"]
        if requested_metrics is None:
                requested_metrics = []

        if mode == "train":
                net.train()
                desc = f">> Training   ({epc})"
        elif mode == "val":
                net.eval()
                desc = f">> Validation ({epc})"
        elif mode == "dual":
                net.eval()
                desc = f">> Dual       ({epc})"

        total_iteration: int = sum(len(loader) for loader in loaders)  # U
        total_images: int = sum(len(loader.dataset) for loader in loaders)  # D
        n_loss: int = max(map(len, list_loss_fns))

        epoch_metrics: dict[str, Tensor]
        epoch_metrics = {"dice": torch.zeros((total_images, K), dtype=torch.float32, device=device),
                         "loss": torch.zeros((total_iteration, n_loss), dtype=torch.float32, device=device)}

        if "3d_dsc" in requested_metrics:
                epoch_metrics["3d_dsc"] = torch.zeros((total_iteration, K), dtype=torch.float32, device=device)

        few_axis: bool = len(metric_axis) <= 4

        # time_log: np.ndarray = np.ndarray(total_iteration, dtype=np.float32)

        done_img: int = 0
        done_batch: int = 0
        tq_iter = tqdm_(total=total_iteration, desc=desc)
        for i, (loader, loss_fns, loss_weights) in enumerate(zip(loaders, list_loss_fns, list_loss_weights)):
                for data in loader:
                        # t0 = time()
                        image: Tensor = data["images"].to(device)
                        target: Tensor = data["gt"].to(device)
                        filenames: list[str] = data["filenames"]
                        assert not target.requires_grad

                        labels: list[Tensor] = [e.to(device) for e in data["labels"]]
                        bounds: list[Tensor] = [e.to(device) for e in data["bounds"]]
                        assert len(labels) == len(bounds)

                        B, C, *_ = image.shape

                        samplings: list[list[Tuple[slice]]] = data["samplings"]
                        assert len(samplings) == B
                        assert len(samplings[0][0]) == len(image[0, 0].shape), (samplings[0][0], image[0, 0].shape)

                        probs_receptacle: Tensor = - torch.ones_like(target, dtype=torch.float32)  # -1 for unfilled
                        mask_receptacle: Tensor = - torch.ones_like(target, dtype=torch.int32)  # -1 for unfilled

                        # Use the sampling coordinates of the first batch item
                        assert not (len(samplings[0]) > 1 and B > 1), samplings  # No subsampling if batch size > 1
                        loss_sub_log: Tensor = torch.zeros((len(samplings[0]), len(loss_fns)), 
                                                           dtype=torch.float32, device=device)
                        for k, sampling in enumerate(samplings[0]):
                                img_sampling = [slice(0, B), slice(0, C)] + list(sampling)
                                label_sampling = [slice(0, B), slice(0, K)] + list(sampling)
                                assert len(img_sampling) == len(image.shape), (img_sampling, image.shape)
                                sub_img = image[img_sampling]

                                # Reset gradients
                                if optimizer:
                                        optimizer.zero_grad()

                                # Forward
                                pred_logits: Tensor = net(sub_img)
                                pred_probs: Tensor = F.softmax(temperature * pred_logits, dim=1)

                                # Used only for dice computation:
                                predicted_mask: Tensor = probs2one_hot(pred_probs.detach())  
                                assert not predicted_mask.requires_grad

                                probs_receptacle[label_sampling] = pred_probs[...]
                                mask_receptacle[label_sampling] = predicted_mask[...]

                                assert len(bounds) == len(loss_fns) == len(loss_weights) == len(labels)
                                ziped = zip(loss_fns, labels, loss_weights, bounds)
                                losses = [w * loss_fn(pred_probs, label[label_sampling], bound, filenames)
                                          for loss_fn, label, w, bound in ziped]
                                loss = reduce(add, losses)
                                assert loss.shape == (), loss.shape

                                # Backward
                                if optimizer:
                                        loss.backward()
                                        optimizer.step()

                                # Compute and log metrics
                                for j in range(len(loss_fns)):
                                        loss_sub_log[k, j] = losses[j].detach()
                        reduced_loss_sublog: Tensor = loss_sub_log.sum(dim=0)
                        assert reduced_loss_sublog.shape == (len(loss_fns),), (reduced_loss_sublog.shape, len(loss_fns))
                        epoch_metrics["loss"][done_batch, ...] = reduced_loss_sublog[...]
                        del loss_sub_log

                        sm_slice = slice(done_img, done_img + B)  # Values only for current batch

                        dices: Tensor = dice_coef(mask_receptacle, target)
                        assert dices.shape == (B, K), (dices.shape, B, K)
                        epoch_metrics["dice"][sm_slice, ...] = dices

                        if "3d_dsc" in requested_metrics:
                                three_d_DSC: Tensor = dice_batch(mask_receptacle, target)
                                assert three_d_DSC.shape == (K,)

                                epoch_metrics["3d_dsc"][done_batch] = three_d_DSC  # type: ignore

                        # Save images
                        if savedir:
                                with warnings.catch_warnings():
                                        warnings.filterwarnings("ignore", category=UserWarning)
                                        predicted_class: Tensor = probs2class(pred_probs)
                                        save_images(predicted_class, 
                                                    data["filenames"], 
                                                    savedir / f"iter{epc:03d}" / mode)

                        # Logging
                        big_slice = slice(0, done_img + B)  # Value for current and previous batches

                        stat_dict: dict[str, Any] = {}
                        # The order matters for the final display -- it is easy to change

                        if few_axis:
                                stat_dict |= {f"DSC{n}": epoch_metrics["dice"][big_slice, n].mean()
                                              for n in metric_axis}

                                if "3d_dsc" in requested_metrics:
                                        stat_dict |= {f"3d_DSC{n}": epoch_metrics["3d_dsc"][:done_batch, n].mean()
                                                      for n in metric_axis}

                        if len(metric_axis) > 1:
                                stat_dict |= {"DSC": epoch_metrics["dice"][big_slice, metric_axis].mean()}

                        stat_dict |= {f"loss_{i}": epoch_metrics["loss"][:done_batch].mean(dim=0)[i] 
                                      for i in range(n_loss)}

                        nice_dict = {k: f"{v:.3f}" for (k, v) in stat_dict.items()}

                        # t1 = time()
                        # time_log[done_batch] = (t1 - t0)

                        done_img += B
                        done_batch += 1
                        tq_iter.set_postfix({**nice_dict, "loader": str(i)})
                        tq_iter.update(1)
        tq_iter.close()

        print(f"{desc} " + ', '.join(f"{k}={v}" for (k, v) in nice_dict.items()))

        return {k: v.detach().cpu() for (k, v) in epoch_metrics.items()}


def run(args: argparse.Namespace) -> dict[str, Tensor]:
        n_class: int = args.n_class
        lr: float = args.l_rate
        savedir: Path = Path(args.workdir)
        n_epoch: int = args.n_epoch
        val_f: int = args.val_loader_id

        loss_fns: list[list[Callable]]
        loss_weights: list[list[float]]
        net, optimizer, device, loss_fns, loss_weights, scheduler = setup(args, n_class)
        train_loaders: list[DataLoader]
        val_loaders: list[DataLoader]
        train_loaders, val_loaders = get_loaders(args, args.dataset,
                                                 args.batch_size, n_class,
                                                 args.debug, args.in_memory, args.dimensions, args.use_spacing)

        n_tra: int = sum(len(tr_lo.dataset) for tr_lo in train_loaders)  # Number of images in dataset
        l_tra: int = sum(len(tr_lo) for tr_lo in train_loaders)  # Number of iteration per epc: different if batch_size > 1
        n_val: int = sum(len(vl_lo.dataset) for vl_lo in val_loaders)
        l_val: int = sum(len(vl_lo) for vl_lo in val_loaders)
        n_loss: int = max(map(len, loss_fns))

        best_dice: Tensor = torch.tensor(0, dtype=torch.float32)
        best_epoch: int = 0
        metrics: dict[str, Tensor] = {"val_dice": torch.zeros((n_epoch, n_val, n_class), dtype=torch.float32),
                                      "val_loss": torch.zeros((n_epoch, l_val, len(loss_fns[val_f])), dtype=torch.float32),
                                      "tra_dice": torch.zeros((n_epoch, n_tra, n_class), dtype=torch.float32),
                                      "tra_loss": torch.zeros((n_epoch, l_tra, n_loss), dtype=torch.float32)}
        
        if args.compute_3d_dice:
                metrics["val_3d_dsc"] = torch.zeros((n_epoch, l_val, n_class), dtype=torch.float32)
                best_3d_dsc: Tensor = torch.tensor(0, dtype=torch.float32)

        tra_req_metrics: list[str] = [k.removeprefix("tra_") for k in metrics.keys() if "tra_" in k]
        val_req_metrics: list[str] = [k.removeprefix("val_") for k in metrics.keys() if "val_" in k]

        print("\n>>> Starting the training")
        for i in range(n_epoch):
                # Do training and validation loops
                tra_metrics = do_epoch("train", net, device, train_loaders, i,
                                       loss_fns, loss_weights, n_class,
                                       savedir=savedir if args.save_train else None,
                                       optimizer=optimizer,
                                       metric_axis=args.metric_axis,
                                       requested_metrics=tra_req_metrics,
                                       temperature=args.temperature)
                with torch.no_grad():
                        val_metrics = do_epoch("val", net, device, val_loaders, i,
                                               [loss_fns[val_f]],
                                               [loss_weights[val_f]],
                                               n_class,
                                               savedir=savedir,
                                               metric_axis=args.metric_axis,
                                               requested_metrics=val_req_metrics,
                                               temperature=args.temperature)

                # Sort and save the metrics
                for mode, mode_metrics in zip(["tra_", "val_"],
                                              [tra_metrics, val_metrics]):
                        for k in mode_metrics:
                                key: str = f"{mode}{k}"
                                assert metrics[key][i].shape == mode_metrics[k].shape, \
                                    (metrics[key][i].shape, mode_metrics[k].shape, k)
                                metrics[key][i] = mode_metrics[k]

                for k, e in metrics.items():
                        np.save(savedir / f"{k}.npy", e.cpu().numpy())

                df = pd.DataFrame({"tra_loss": metrics["tra_loss"].mean(dim=(1, 2)).numpy(),
                                   "val_loss": metrics["val_loss"].mean(dim=(1, 2)).numpy(),
                                   "tra_dice": metrics["tra_dice"][:, :, -1].mean(dim=1).numpy(),
                                   "val_dice": metrics["val_dice"][:, :, -1].mean(dim=1).numpy()})
                df.to_csv(savedir / args.csv, float_format="%.4f", index_label="epoch")

                # Save model if better
                current_dice: Tensor = metrics["val_dice"][i, :, args.metric_axis].mean()
                if current_dice > best_dice:
                        best_epoch = i
                        best_dice = current_dice
                        if "val_3d_dsc" in metrics:
                                best_3d_dsc = metrics["val_3d_dsc"][i, :, args.metric_axis].mean()

                        with open(savedir / "best_epoch.txt", 'w') as f:
                                f.write(str(i))
                        best_folder = savedir / "best_epoch"
                        if best_folder.exists():
                                rmtree(best_folder)
                        copytree(savedir / f"iter{i:03d}", Path(best_folder))
                        torch.save(net, savedir / "best.pkl")

                optimizer, loss_fns, loss_weights = scheduler(i, optimizer, loss_fns, loss_weights,
                                                              net, device, train_loaders, args)

                # if args.schedule and (i > (best_epoch + 20)):
                if args.schedule and (i % (best_epoch + 20) == 0):  # Yeah, ugly but will clean that later
                        for param_group in optimizer.param_groups:
                                lr *= 0.5
                                param_group['lr'] = lr
                                print(f'>> New learning Rate: {lr}')

                if i > 0 and not (i % 5):
                        maybe_3d = f', 3d_DSC: {best_3d_dsc:.3f}' if args.compute_3d_dice else ''
                        print(f">> Best results at epoch {best_epoch}: DSC: {best_dice:.3f}{maybe_3d}")

        # Because displaying the results at the end is actually convenient:
        maybe_3d = f', 3d_DSC: {best_3d_dsc:.3f}' if args.compute_3d_dice else ''
        print(f">> Best results at epoch {best_epoch}: DSC: {best_dice:.3f}{maybe_3d}")
        for metric in metrics:
                # Do not care about training values, nor the loss (keep it simple):
                if "val" in metric or "loss" in metric:
                        print(f"\t{metric}: {metrics[metric][best_epoch].mean(dim=0)}")

        return metrics


def get_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description='Hyperparams')
        parser.add_argument('--dataset', type=str, required=True)
        parser.add_argument("--csv", type=str, required=True)
        parser.add_argument("--workdir", type=str, required=True)
        parser.add_argument("--losses", type=str, required=True,
                            help="list of list of (loss_name, loss_params, bounds_name, bounds_params, fn, weight)")
        parser.add_argument("--folders", type=str, required=True,
                            help="list of list of (subfolder, transform, is_hot)")
        parser.add_argument("--network", type=str, required=True, help="The network to use")
        parser.add_argument("--n_class", type=int, required=True)
        parser.add_argument("--metric_axis", type=int, nargs='*', required=True, help="Classes to display metrics. \
                Display only the average of everything if empty")

        parser.add_argument("--augment_blur", action="store_true")
        parser.add_argument("--blur_onlyfirst", action="store_true",
                            help="Blur only the first image, i.e., the image, and not the ground truth.")
        parser.add_argument("--augment_rotate", action="store_true")
        parser.add_argument("--augment_scale", action="store_true")
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--cpu", action='store_true')
        parser.add_argument("--in_memory", action='store_true')
        parser.add_argument("--schedule", action='store_true')
        parser.add_argument("--use_sgd", action='store_true')
        parser.add_argument("--compute_3d_dice", action='store_true')
        parser.add_argument("--save_train", action='store_true')
        parser.add_argument("--use_spacing", action='store_true')
        parser.add_argument("--three_d", action='store_true')
        parser.add_argument("--no_assert_dataloader", action='store_true')
        parser.add_argument("--ignore_norm_dataloader", action='store_true')
        parser.add_argument("--group", action='store_true', help="Group the patient slices together for validation. \
                Useful to compute the 3d dice, but might destroy the memory for datasets with a lot of slices per patient.")
        parser.add_argument("--group_train", action='store_true', help="Group the patient slices together for training. \
                Useful to compute the 3d dice, but might destroy the memory for datasets with a lot of slices per patient.")

        parser.add_argument('--n_epoch', nargs='?', type=int, default=200,
                            help='# of the epochs')
        parser.add_argument('--l_rate', nargs='?', type=float, default=5e-4,
                            help='Learning Rate')
        parser.add_argument("--grp_regex", type=str, default=None)
        parser.add_argument('--temperature', type=float, default=1, help="Temperature for the softmax")
        parser.add_argument("--scheduler", type=str, default="DummyScheduler")
        parser.add_argument("--scheduler_params", type=str, default="{}")
        parser.add_argument("--modalities", type=int, default=1)
        parser.add_argument("--dimensions", type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument("--weights", type=str, default='', help="Stored weights to restore")
        parser.add_argument("--training_folders", type=str, nargs="+", default=["train"])
        parser.add_argument("--validation_folder", type=str, default="val")
        parser.add_argument("--val_loader_id", type=int, default=-1, help="""
                Kinda housefiry at the moment. When we have several train loader (for hybrid training
                for instance), wants only one validation loader. The way the dataloading creation is
                written at the moment, it will create several validation loader on the same topfolder (val),
                but with different folders/bounds ; which will basically duplicate the evaluation.
                """)

        args = parser.parse_args()
        if args.metric_axis == []:
                args.metric_axis = list(range(args.n_class))
        print("\n", args)

        return args


if __name__ == '__main__':
        run(get_args())
