CC = python3.9
PP = PYTHONPATH="$(PYTHONPATH):."
SHELL = zsh


.PHONY: all metrics plot view view_labels pack report train weak

red:=$(shell tput bold ; tput setaf 1)
green:=$(shell tput bold ; tput setaf 2)
yellow:=$(shell tput bold ; tput setaf 3)
blue:=$(shell tput bold ; tput setaf 4)
magenta:=$(shell tput bold ; tput setaf 5)
cyan:=$(shell tput bold ; tput setaf 6)
reset:=$(shell tput sgr0)

# RD stands for Result DIR -- useful way to report from extracted archive
RD = results/promise12

# CFLAGS = -O
# DEBUG = --debug
EPC = 300
BS = 4  # BS stands for Batch Size
LR = 5e-4  # Learning rate
K = 2  # K for class

G_RGX = (Case\d+_\d+)_\d+
B_DATA = [('img', png_transform, False), ('gt', gt_transform, True)]
NET = ResidualUNet
# NET = Dummy


TRN = $(RD)/ceaugment \
	$(RD)/oursaugment


GRAPH = $(RD)/val_dice.png $(RD)/tra_dice.png \
		$(RD)/tra_loss.png \
	$(RD)/val_3d_dsc.png 
BOXPLOT = $(RD)/val_3d_dsc_boxplot.png $(RD)/test_3d_dsc_boxplot.png
PLT = $(GRAPH) $(BOXPLOT)

REPO = $(shell basename `git rev-parse --show-toplevel`)
DATE = $(shell date +"%y%m%d")
HASH = $(shell git rev-parse --short HEAD)
HOSTNAME = $(shell hostname)
PBASE = archives
PACK = $(PBASE)/$(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-$(shell basename $(RD)).tar.zst

all: pack

train: $(TRN)
plot: $(PLT)

pack: $(PACK) report
$(PACK): $(PLT) $(TRN)
	$(info $(red)tar cf $@$(reset))
	mkdir -p $(@D)
	tar cf - $^ | zstd -T0 -3 > $@
	chmod -w $@
# 	tar cf $@ - $^
# 	tar cf - $^ | pigz > $@
# 	tar cvf - $^ | zstd -T0 -#3 > $@
# tar -zc -f $@ $^  # Use if pigz is not available


# Data generation
data/PROMISE12/train/gt data/PROMISE12/val/gt: data/PROMISE12
data/PROMISE12:  OPT = --seed=0 --retains 5 --retains_test 10
data/PROMISE12: data/promise12
	rm -rf $@_tmp
	$(PP) $(CC) $(CFLAGS) preprocess/slice_promise.py --source_dir $< --dest_dir $@_tmp $(OPT)
	mv $@_tmp $@
data/promise12: data/prostate.lineage data/TrainingData_Part1.zip data/TrainingData_Part2.zip data/TrainingData_Part3.zip
	md5sum -c $<
	rm -rf $@_tmp
	unzip -q $(word 2, $^) -d $@_tmp
	unzip -q $(word 3, $^) -d $@_tmp
	unzip -q $(word 4, $^) -d $@_tmp
	mv $@_tmp $@


data/PROMISE12/train/img data/PROMISE12/val/img: | data/PROMISE12
data/PROMISE12/train/gt data/PROMISE12/val/gt: | data/PROMISE12

data/PROMISE12/train/img data/PROMISE12/val/img data/PROMISE12/test/img: | data/PROMISE12
data/PROMISE12/train/gt data/PROMISE12/val/gt data/PROMISE12/test/gt: | data/PROMISE12


# Trainings
## Full ones
$(RD)/ceaugment: OPT = --losses="[('CrossEntropy', {'idc': [0, 1]}, None, None, None, 1)]" \
	--augment_blur --blur_onlyfirst --augment_rotate --augment_scale
$(RD)/ceaugment: data/PROMISE12/train/gt data/PROMISE12/val/gt
$(RD)/ceaugment: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True)]"


## Beyond pixel-wise supervision
$(RD)/oursaugment: OPT = --losses="[('LogBarrierLoss', {'idc': [0, 1], 't': 1}, 'PreciseBounds', {'margin': 0.10, 'mode': 'percentage'}, 'soft_size', 1e-2), \
	('LogBarrierLoss', {'idc': [1], 't': 1}, 'PreciseBounds', {'margin': 20, 'mode': 'abs'}, 'soft_centroid', 1e-2), \
	('LogBarrierLoss', {'idc': [1], 't': 1}, 'PreciseBounds', {'margin': 20, 'mode': 'abs'}, 'soft_dist_centroid', 1e-2), \
	('LogBarrierLoss', {'idc': [1], 't': 1}, 'PreciseBounds', {'margin': 10, 'mode': 'percentage'}, 'soft_length', 1e-2)]" \
	--scheduler=MultiplyT --scheduler_params="{'target_loss': 'LogBarrierLoss', 'mu': 1.1}" \
	--augment_blur --blur_onlyfirst --augment_rotate --augment_scale
$(RD)/oursaugment: data/PROMISE12/train/gt data/PROMISE12/val/gt
$(RD)/oursaugment: DATA = --folders="$(B_DATA)+[('gt', gt_transform, True), ('gt', gt_transform, True), ('gt', gt_transform, True), ('gt', gt_transform, True)]"



# Template
$(RD)/%:
	$(info $(green)$(CC) $(CFLAGS) main.py $@$(reset))
	rm -rf $@_tmp
	mkdir -p $@_tmp
	printenv > $@_tmp/env.txt
	git diff > $@_tmp/repo.diff
	git rev-parse --short HEAD > $@_tmp/commit_hash
	-$(CC) $(CFLAGS) main.py --dataset=$(dir $(<D)) --batch_size=$(BS) --group --schedule \
		--n_epoch=$(EPC) --workdir=$@_tmp --csv=metrics.csv --n_class=$(K) --metric_axis 1 \
		--l_rate=$(LR) \
		--compute_3d_dice \
		--grp_regex="$(G_RGX)" --network=$(NET) $(OPT) $(DATA) $(DEBUG)
	mv $@_tmp $@


# Inference
$(RD)/%/best_epoch/test: data/PROMISE12/test/img | $(RD)/%
	$(info $(magenta)$(CC) $(CFLAGS) inference.py $@$(reset))
	rm -rf $@_tmp
	mkdir -p $@_tmp
	$(CC) $(CFLAGS) inference.py --data_folder $< --save_folder $@ \
		--model_weights $(dir $(@D))/best.pkl \
		--num_classes 4 --mode argmax
	mv $@_tmp $@


# Metrics
METRICS = $(addsuffix /test_3d_dsc.npy, $(TRN))
metrics: $(METRICS) | $(addsuffix /best_epoch/test, $(TRN))

$(RD)/%/test_3d_dsc.npy: $(RD)/%/best_epoch/test | data/PROMISE12/test/gt
	$(info $(cyan)$(CC) $(CFLAGS) metrics.py $@$(reset))
	$(CC) $(CFLAGS) metrics.py --pred_folders $< --pred_transforms gt_transform \
		--extensions .png .png --gt_folder data/PROMISE12/test/gt \
		--grp_regex="$(G_RGX)" --num_classes=$(K) \
		--save_folder=$(@D) --mode test \
		--metrics "3d_dsc"




# Plotting
$(RD)/val_3d_dsc.png $(RD)/val_dice.png $(RD)/tra_dice.png: COLS = 1
$(RD)/tra_loss.png: COLS = 0
$(RD)/tra_loss.png: OPT = --dynamic_third_axis
$(RD)/val_dice.png $(RD)/tra_loss.png $(RD)/val_3d_dsc.png: plot.py $(TRN)
$(RD)/tra_dice.png: plot.py $(TRN)

$(RD)/val_3d_dsc_boxplot.png: COLS = 1
$(RD)/val_3d_dsc_boxplot.png: moustache.py $(TRN)

$(RD)/test_3d_dsc_boxplot.png: COLS = 1
$(RD)/test_3d_dsc_boxplot.png: moustache.py $(TRN) | metrics

$(RD)/%.png:
	$(info $(blue)$(CC) $(CFLAGS) $< $@$(reset))
	$(eval metric:=$(subst _hist,,$(@F)))
	$(eval metric:=$(subst _boxplot,,$(metric)))
	$(eval metric:=$(subst .png,.npy,$(metric)))
	$(CC) $(CFLAGS) $< --filename $(metric) --folders $(filter-out $<,$^) --columns $(COLS) \
		--savefig=$@ --headless --epc 199 $(OPT)


# Viewing
view: $(TRN) | weak
	$(info $(cyan)$(CC) viewer/viewer.py $^ $(reset))
	$(CC) viewer/viewer.py -n 3 --img_source data/PROMISE12/val/img data/PROMISE12/val/gt \
		$(addsuffix /best_epoch/val, $^) --crop 10 \
		--display_names gt $(notdir $^) --no_contour -C $(K)

view_labels: data/PROMISE12/train/gt
	$(info $(cyan)$(CC) viewer/viewer.py $^ $(reset))
	$(CC) viewer/viewer.py -n 3 --img_source data/PROMISE12/val/img $^ --crop 10 \
		--display_names $(notdir $^) --no_contour -C $(K)

report: $(TRN) | metrics
	$(info $(yellow)$(CC) $(CFLAGS) report.py$(reset))
	$(CC) $(CFLAGS) report.py --folders $(TRN) --metrics val_3d_dsc val_dice --axises 1 $(DEBUG)
	$(CC) $(CFLAGS) report.py --folders $(TRN) --metrics test_3d_dsc --axises 1 $(DEBUG)

test_augments: data/PROMISE12/train
	$(info $(yellow)$(CC) $(CFLAGS) test_augment.py$(reset))
	$(CC) $(CFLAGS) test_augment.py --data_root $< --num_classes $(K)


MIDL2021 = $(RD)/ceaugment $(RD)/oursaugment

midl2021: $(MIDL2021) \
	$(RD)/midl2021/tra_dice.png \
	$(RD)/midl2021/val_3d_dsc.png \
	$(RD)/midl2021/test_3d_dsc_boxplot.png

$(RD)/midl2021/tra_dice.png: plot.py $(MIDL2021)
$(RD)/midl2021/tra_dice.png: COLS = 1
$(RD)/midl2021/tra_dice.png: OPT = --title "Promise12 training DSC over time" \
	--labels "Cross-entropy" "Shape descriptors" \
	--ylabel "DSC" \
	--figsize 10 8 --fontsize 22 --loc "lower right"

$(RD)/midl2021/val_3d_dsc.png: plot.py $(MIDL2021)
$(RD)/midl2021/val_3d_dsc.png: COLS = 1
$(RD)/midl2021/val_3d_dsc.png: OPT = --title "Promise12 validation DSC over time" \
	--labels "Cross-entropy" "Shape descriptors" \
	--ylabel "DSC" \
	--figsize 10 8 --fontsize 22 --loc "lower right"

$(RD)/midl2021/test_3d_dsc_boxplot.png: moustache.py $(MIDL2021) | metrics
$(RD)/midl2021/test_3d_dsc_boxplot.png: COLS = 1
$(RD)/midl2021/test_3d_dsc_boxplot.png: OPT = --title "Promise12 testing DSC" \
	--labels "Cross-entropy" "Shape descriptors" \
	--ylabel "DSC" --xlabel " " \
	--figsize 10 10 --fontsize 22


$(RD)/midl2021/%.png:
	$(info $(blue)$(CC) $(CFLAGS) $< $@$(reset))
	$(eval metric:=$(subst _hist,,$(@F)))
	$(eval metric:=$(subst _boxplot,,$(metric)))
	$(eval metric:=$(subst .png,.npy,$(metric)))
	mkdir -p $(@D)
	$(CC) $(CFLAGS) $< --filename $(metric) --folders $(filter-out $<,$^) --columns $(COLS) \
		--savefig=$@ --headless --epc 199 $(OPT)
