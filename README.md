# Beyond pixel-wise supervision
Official repository for [Beyond pixel-wise supervision for segmentation: A few global shape descriptors might be surprisingly good!](https://openreview.net/forum?id=nqe6e0oJ_fL), currently under review at [MIDL 2021](https://2021.midl.io). 


## Table of contents
* [Table of contents](#table-of-contents)
* [Requirements (PyTorch)](#requirements-pytorch)
* [Usage](#usage)
* [Automation](#automation)
    * [Data scheme](#data-scheme)
        * [dataset](#dataset)
        * [results](#results)
    * [Cool tricks](#cool-tricks)

## Requirements (PyTorch)
To reproduce our experiments:
* python3.9
* Pytorch 1.7
* nibabel (only for slicing)
* Scipy
* NumPy
* Matplotlib
* Scikit-image
* zsh

## Usage
Shape descriptors are defined in `utils.py`, and the log-barrier are defined in the `losses.py` file. Examples to combine the two are available in the two makefiles, `acdc.make` and `promise12.make`. For instance:
```make
$(RD)/oursaugment: OPT = --losses="[('LogBarrierLoss', {'idc': [0, 1], 't': 1}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 1e-2), \
    ('LogBarrierLoss', {'idc': [1], 't': 1}, 'PreciseBounds', {'margin': 20, 'mode': 'abs'}, 'soft_centroid', 1e-2), \
    ('LogBarrierLoss', {'idc': [1], 't': 1}, 'PreciseBounds', {'margin': 20, 'mode': 'abs'}, 'soft_dist_centroid', 1e-2), \
    ('LogBarrierLoss', {'idc': [1], 't': 1}, 'PreciseBounds', {'margin': 10, 'mode': 'percentage'}, 'soft_length', 1e-2)]" \
    --scheduler=MultiplyT --scheduler_params="{'target_loss': 'LogBarrierLoss', 'mu': 1.1}" \
    --augment_blur --blur_onlyfirst --augment_rotate --augment_scale
$(RD)/oursaugment: data/PROSTATE-CL/train/gt data/PROSTATE-CL/val/gt
$(RD)/oursaugment: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True), ('gt', gt_transform, True), ('gt', gt_transform, True), ('gt', gt_transform, True)]"
```
Here, we have a list of 4 losses in total, each a `LogBarrierLoss`, constraining a different function (`soft_size`, `soft_centroid`, `soft_dist_centroid`, `soft_length` respectively). They all share the same weight (`1e-2`) and the same `t`, controlled by the `MultiplyT` scheduler. The `PreciseBounds` compute the lower and upper bounds fed to the constraining function (the LogBarrier extension), plus minus, either, 10%, or 20 pixels (when applicable).


## Automation
Experiments are handled by [GNU Make](https://en.wikipedia.org/wiki/Make_(software)). It should be installed on pretty much any machine.

Instruction to download the data are contained in the lineage files [prostate.lineage](data/prostate.lineage) and [acdc.lineage](data/acdc.lineage). They are text files containing the md5sum of the original zip.

Once the zip is in place, everything should be automatic:
```sh
make -f promise12.make
make -f acdc.make
```
Usually takes a little bit more than a day per makefile.

This perform in the following order:
* unpacking of the data;
* remove unwanted big files;
* normalization and slicing of the data;
* training with the different methods;
* plotting of the metrics curves;
* display of a report;
* archiving of the results in an .tar.gz stored in the `archives` folder.

Make will handle by itself the dependencies between the different parts. For instance, once the data has been pre-processed, it won't do it another time, even if you delete the training results. It is also a good way to avoid overwriting existing results by accident.

Of course, parts can be launched separately :
```sh
make -f promise12.make data/prostate # Unpack only
make -f promise12.make data/prostate # unpack if needed, then slice the data
make -f promise12.make results/promise12/box_prior_box_size_neg_size # train only that setting. Create the data if needed
make -f promise12.make results/promise12/val_dice.png # Create only this plot. Do the trainings if needed
```
There is many options for the main script, because I use the same code-base for other projects. You can safely ignore most of them, and the different recipe in the makefiles should give you an idea on how to modify the training settings and create new targets. In case of questions, feel free to contact me.

The recipes for the MIDL submission plots can be access through:
```sh
make -f promise12.make midl21
make -f acdc.make midl21
```

### Data scheme
#### datasets
For instance
```
promise12/
    train/
        img/
            ...
        gt/
    val/
        img/
        gt/
        ...
```
The network takes npy files as an input (there is multiple modalities), but images for each modality are saved for convenience. The gt folder contains gray-scale images of the ground-truth, where the gray-scale level are the number of the class (namely, 0 and 1). This is because I often use my [segmentation viewer](https://github.com/HKervadec/segmentation_viewer) to visualize the results, so that does not really matter. If you want to see it directly in an image viewer, you can either use the remap script, or use imagemagick:
```
mogrify -normalize data/prostate/val/gt/*.png
```

#### results
```
results/
    promise12/
        fs/
            best_epoch/
                val/
                    Case10_0_0.png
                    ...
            iter000/
                val/
            ...
            best.pkl # best model saved
            metrics.csv # metrics over time, csv
            best_epoch.txt # number of the best epoch
            val_dice.npy # log of all the metric over time for each image and class
        box_prior_box_size_neg_size/
            ...
        val_dice.png # Plot over time comparing different methods
        ...
    acdc/
        ...
archives/
    $(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-promise12.tar.gz
    $(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-acdc.tar.gz
```

### Cool tricks
Remove all assertions from the code. Usually done after making sure it does not crash for one complete epoch:
```sh
make -f promise12.make <anything really> CFLAGS=-O
```

Use a specific python executable:
```sh
make -f promise12.make <super target> CC=/path/to/the/executable
```

Train for only 5 epochs, with a dummy network, and only 10 images per data loader. Useful for debugging:
```sh
make -f promise12.make <really> NET=Dimwit EPC=5 DEBUG=--debug
```

Rebuild everything even if already exist:
```sh
make -f promise12.make <a> -B
```

Only print the commands that will be run:
```sh
make -f promise12.make <a> -n
```

Create a gif for the predictions over time of a specific patient:
```
cd results/promise12/fs
convert iter*/val/Case14_0_0.png Case14_0_0.gif
mogrify -normalize Case14_0_0.gif
```
