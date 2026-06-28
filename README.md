# MDRA-YOLOv8 Experiments — Phase 6.1

This directory currently implements only environment verification, M3FD data auditing, fixed/unified splits, dataset statistics, tiny/small smoke subsets, and the YOLOv8s Visible baseline. It does **not** modify the YOLOv8 architecture.

## 1. Linux environment

AutoDL and similar GPU-cloud images usually provide Ubuntu, CUDA, and PyTorch. Verify the actual image before installing packages:

```bash
uname -a
nvidia-smi
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If the server already contains a working CUDA-enabled PyTorch installation, do not reinstall `torch`/`torchvision` blindly. Install the remaining requirements in the existing environment, or remove those two lines from a temporary requirements copy:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run tests:

```bash
python -m pytest -q
```

## 2. Required path variables

Paths are runtime inputs; the code does not depend on `/root/autodl-tmp` or any Windows drive.

```bash
export PROJECT_ROOT=/path/to/mdra_yolov8_experiments
export DATA_ROOT=/path/to/datasets/M3FD
export OUTPUT_ROOT=/path/to/outputs/mdra_yolov8
export CUDA_VISIBLE_DEVICES=0

cd "$PROJECT_ROOT"
```

All commands below assume they are executed from `$PROJECT_ROOT`.

## 3. M3FD directory structure to confirm manually

The scripts pair files by the path relative to each modality directory, without the extension. For example, these files share sample ID `scene_a/000001`:

```text
$DATA_ROOT/visible/scene_a/000001.jpg
$DATA_ROOT/infrared/scene_a/000001.png
$DATA_ROOT/labels/scene_a/000001.txt
```

A flat layout also works:

```text
M3FD/
├── visible/
│   ├── 000001.jpg
│   └── 000002.jpg
├── infrared/
│   ├── 000001.png
│   └── 000002.png
└── labels/
    ├── 000001.txt
    └── 000002.txt
```

Auto-detection recognizes common names such as `visible`, `VIS`, `rgb`, `infrared`, `IR`, `thermal`, and `labels`. If the local dataset uses other names, pass `--vis-dir`, `--ir-dir`, and `--label-dir` explicitly. Paths may be absolute or relative to `--data-root`.

Labels must use YOLO detection format:

```text
class_id x_center y_center width height
```

All four box values are normalized. The checker also verifies that the derived box corners remain within the image.

## 4. Phase 6.1 run order

### 4.1 P0-ENV

```bash
python scripts/check_env.py \
  --output-root "$OUTPUT_ROOT"
```

Outputs:

```text
$OUTPUT_ROOT/env_reports/env_report_YYYYMMDD_HHMMSS.txt
$OUTPUT_ROOT/env_reports/env_report_YYYYMMDD_HHMMSS.json
```

### 4.2 P0-DATA

```bash
python scripts/check_m3fd_pairs.py \
  --data-root "$DATA_ROOT" \
  --vis-dir visible \
  --ir-dir infrared \
  --label-dir labels \
  --output-root "$OUTPUT_ROOT" \
  --fail-on-error
```

The report distinguishes missing VIS/IR/labels, duplicate stems, unreadable images, size mismatches, invalid YOLO rows, out-of-range fields, and boxes whose corners exceed the image.

Reports are not overwritten by default. Use `--overwrite` only after preserving or intentionally discarding the previous audit.

### 4.3 P0-FIXED-SPLIT

This creates a reproducible **fixed/unified split**, not an official split:

```bash
python scripts/make_fixed_split.py \
  --data-root "$DATA_ROOT" \
  --vis-dir visible \
  --ir-dir infrared \
  --label-dir labels \
  --output-dir configs/splits/m3fd_seed42 \
  --train-ratio 0.7 \
  --val-ratio 0.2 \
  --test-ratio 0.1 \
  --seed 42
```

The command refuses to split invalid data by default and refuses to overwrite an existing split. `--allow-invalid` and `--overwrite` must be explicit.

### 4.4 P0-STATS

```bash
python scripts/dataset_stats.py \
  --data-root "$DATA_ROOT" \
  --split-dir configs/splits/m3fd_seed42 \
  --vis-dir visible \
  --label-dir labels \
  --output-root "$OUTPUT_ROOT"
```

Default small/medium/large thresholds are COCO pixel areas (`32²`, `96²`) computed in original-image coordinates. Override with `--small-area` and `--medium-area`.

Scene content is never guessed from pixels. Without an explicit metadata CSV containing `sample_id,scene`, all samples are reported as `unknown`. To use verified metadata:

```bash
python scripts/dataset_stats.py \
  --data-root "$DATA_ROOT" \
  --split-dir configs/splits/m3fd_seed42 \
  --label-dir labels \
  --scene-metadata /path/to/verified_scene_metadata.csv \
  --output-root "$OUTPUT_ROOT"
```

### 4.5 P0-TINY-SMOKE

Create a deterministic tiny subset:

```bash
python scripts/create_tiny_subset.py \
  --split-dir configs/splits/m3fd_seed42 \
  --output-dir configs/splits/m3fd_tiny_seed42 \
  --num-train 32 \
  --num-val 16 \
  --seed 42
```

Run 1–2 epochs. The command-line values override YAML values:

```bash
python scripts/train_visible.py \
  --config configs/experiments/B1_visible_tiny_smoke.yaml \
  --data-root "$DATA_ROOT" \
  --vis-dir visible \
  --label-dir labels \
  --split-dir configs/splits/m3fd_tiny_seed42 \
  --output-root "$OUTPUT_ROOT" \
  --epochs 2 \
  --batch 4 \
  --imgsz 640 \
  --device 0
```

Success means forward/backward, validation, checkpoint saving, `results.csv`, and one validation-image prediction all complete.

### 4.6 P0-SMALL-TRAIN

```bash
python scripts/create_tiny_subset.py \
  --split-dir configs/splits/m3fd_seed42 \
  --output-dir configs/splits/m3fd_small_seed42 \
  --num-train 300 \
  --num-val 100 \
  --seed 42

python scripts/train_visible.py \
  --config configs/experiments/B1_visible_small_train.yaml \
  --data-root "$DATA_ROOT" \
  --vis-dir visible \
  --label-dir labels \
  --split-dir configs/splits/m3fd_small_seed42 \
  --output-root "$OUTPUT_ROOT" \
  --epochs 10 \
  --batch 8 \
  --imgsz 640 \
  --device 0
```

If the selected fixed split contains fewer samples than requested, reduce `--num-train` or `--num-val`; the subset script fails instead of silently reusing samples.

## 5. Visible training behavior

`train_visible.py` creates a unique output directory and a standard Ultralytics dataset view:

```text
$OUTPUT_ROOT/experiments/<experiment_id>/
├── resolved_config.yaml
├── environment.json
├── environment.txt
├── command.txt
├── train_arguments.json
├── dataset_view/
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   ├── data.yaml
│   └── manifest.json
├── weights/{best,last}.pt
├── results.csv
├── predictions/single_batch/
└── run_summary.json
```

The dataset view uses symbolic links by default, so source files are not copied or modified. If symlinks are unavailable, pass `--link-mode hardlink` or `--link-mode copy`.

Every new run receives a unique directory. A true Ultralytics resume is the deliberate exception because it continues the original run:

```bash
python scripts/train_visible.py \
  --config configs/experiments/B1_visible_small_train.yaml \
  --data-root "$DATA_ROOT" \
  --split-dir configs/splits/m3fd_small_seed42 \
  --output-root "$OUTPUT_ROOT" \
  --resume "$OUTPUT_ROOT/experiments/B1_visible_small_train/weights/last.pt"
```

The resume request and environment are saved under `$OUTPUT_ROOT/resume_logs/`.

## 6. Output safety

- Pair checks, statistics, and split generation refuse to overwrite files unless `--overwrite` is passed.
- Training always creates a unique run directory.
- Original M3FD images and labels are read-only inputs.
- Split files store sample IDs, not machine-specific absolute paths.

## 7. Current scope

Implemented:

- P0-ENV;
- P0-DATA;
- P0-FIXED-SPLIT;
- P0-STATS;
- P0-TINY-SMOKE;
- P0-SMALL-TRAIN;
- YOLOv8s-Visible wrapper.

Not implemented in this phase:

- IR training;
- EarlyFusion;
- LCMF;
- P2;
- DRA or validation-time DRA diagnostics;
- SR-style auxiliary;
- SegAux;
- ECA;
- SOTA comparisons;
- second datasets.

