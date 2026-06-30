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
export PROJECT_ROOT=/root/autodl-tmp/mdra_yolov8_experiments
export DATA_ROOT=/root/autodl-tmp/M3FD_Detection
export OUTPUT_ROOT="$PROJECT_ROOT/outputs"
export CUDA_VISIBLE_DEVICES=0

cd "$PROJECT_ROOT"
```

All commands below assume they are executed from `$PROJECT_ROOT`.

## 3. M3FD directory structure to confirm manually

The scripts pair files by the path relative to each modality directory, without the extension. For example, these files share sample ID `scene_a/000001`:

```text
$DATA_ROOT/Vis/scene_a/000001.jpg
$DATA_ROOT/Ir/scene_a/000001.png
$DATA_ROOT/labels/scene_a/000001.txt
```

A flat layout also works:

```text
M3FD_Detection/
├── Vis/
│   ├── 000001.jpg
│   └── 000002.jpg
├── Ir/
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
  --vis-dir Vis \
  --ir-dir Ir \
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
  --vis-dir Vis \
  --ir-dir Ir \
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
  --vis-dir Vis \
  --label-dir labels \
  --output-root "$OUTPUT_ROOT"
```

Default small/medium/large thresholds are COCO pixel areas (`32²`, `96²`) computed in original-image coordinates. Override with `--small-area` and `--medium-area`.

Scene content is never guessed from pixels. Without an explicit metadata CSV containing `sample_id,scene`, all samples are reported as `unknown`. To use verified metadata:

```bash
python scripts/dataset_stats.py \
  --data-root "$DATA_ROOT" \
  --split-dir configs/splits/m3fd_seed42 \
  --vis-dir Vis \
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
  --vis-dir Vis \
  --label-dir labels \
  --split-dir configs/splits/m3fd_tiny_seed42 \
  --output-root "$OUTPUT_ROOT" \
  --model "$PROJECT_ROOT/weights/yolov8s.pt" \
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
  --vis-dir Vis \
  --label-dir labels \
  --split-dir configs/splits/m3fd_small_seed42 \
  --output-root "$OUTPUT_ROOT" \
  --model "$PROJECT_ROOT/weights/yolov8s.pt" \
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

## 6. 6.2 B1–B5 baseline code

The 6.2 code uses one paired-data/training/evaluation protocol for all variants:

| Config | Variant | Input / detection scales |
| --- | --- | --- |
| `B1_visible.yaml` | Visible | RGB, P3/P4/P5 |
| `B2_infrared.yaml` | Infrared | grayscale repeated to 3 channels, P3/P4/P5 |
| `B3_early_fusion.yaml` | EarlyFusion | RGB+IR 4-channel input, P3/P4/P5 |
| `B3_early_fusion_p2.yaml` | EarlyFusion-P2 | RGB+IR 4-channel input, P2/P3/P4/P5 |
| `B4_lcmf.yaml` | LCMF | dual stream; `Concat -> 1x1 Conv -> BN -> SiLU`, P3/P4/P5 |
| `B5_lcmf_p2.yaml` | LCMF-P2 | same LCMF with P2/P3/P4/P5 detection |

LCMF and P2 are framework/auxiliary baselines, not the paper's core contribution. No DRA, MESA, SR-style auxiliary, SegAux, or ECA is included in these configs.

### 6.1 GPU preflight after switching to GPU mode

Do not run formal training before all five commands pass. Each command reads two real paired samples and performs one forward, detection loss, and backward pass; it does not run an epoch or update weights.

```bash
for CONFIG in \
  configs/experiments/B1_visible.yaml \
  configs/experiments/B2_infrared.yaml \
  configs/experiments/B3_early_fusion.yaml \
  configs/experiments/B4_lcmf.yaml \
  configs/experiments/B5_lcmf_p2.yaml
do
  python scripts/train_b_baseline.py \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT" \
    --vis-dir Vis \
    --ir-dir Ir \
    --label-dir labels \
    --split-dir configs/splits/m3fd_seed42 \
    --output-root "$OUTPUT_ROOT" \
    --device 0 \
    --dry-run || exit 1
done
```

Reports are saved under `$OUTPUT_ROOT/b_dry_runs/`.

For an unattended sequential run with compact monitoring, use the controller after all five preflights pass:

```bash
nohup bash scripts/run_b_suite.sh \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --split-dir configs/splits/m3fd_seed42 \
  --device 0 \
  --skip-preflight \
  > "$OUTPUT_ROOT/b_suite_launcher.log" 2>&1 &
```

The controller runs B1 through B5 sequentially and stops on the first failure. After each training run it performs fixed-test evaluation, saves 20 prediction visualizations, and profiles Params/GFLOPs/latency/FPS. Monitoring only needs the compact files below; full logs remain on the server:

```bash
CONTROLLER_DIR="$(cat "$OUTPUT_ROOT/b_suite_controller/latest.txt")"
cat "$CONTROLLER_DIR/status.txt"
tail -n 5 "$CONTROLLER_DIR/history.tsv"
```

### 6.2 Formal training order

Run B1 through B5 sequentially so errors and memory usage are known before the larger dual-stream models. The YAML files freeze the common seed, fixed split, image size, optimizer, schedule, and evaluation thresholds. B4/B5 use batch 8 with effective batch 16; B1–B3 use batch 16.

```bash
python scripts/train_b_baseline.py \
  --config configs/experiments/B1_visible.yaml \
  --data-root "$DATA_ROOT" --vis-dir Vis --ir-dir Ir --label-dir labels \
  --split-dir configs/splits/m3fd_seed42 --output-root "$OUTPUT_ROOT" --device 0
```

Replace the config in order with `B2_infrared.yaml`, `B3_early_fusion.yaml`, `B4_lcmf.yaml`, and `B5_lcmf_p2.yaml`. If B4/B5 runs out of memory, reduce `--batch` while keeping `--effective-batch 16`; record the actual batch in the result table.

### 6.3 Validation, prediction, and complexity

```bash
python scripts/validate_b_baseline.py \
  --checkpoint /path/to/run/weights/best.pt \
  --output-root "$OUTPUT_ROOT" --device 0

python scripts/predict_b_baseline.py \
  --checkpoint /path/to/run/weights/best.pt \
  --data-root "$DATA_ROOT" \
  --split-dir configs/splits/m3fd_seed42 \
  --split test --max-images 20 \
  --output-root "$OUTPUT_ROOT" --device 0

python scripts/profile_b_baseline.py \
  --config configs/experiments/B1_visible.yaml \
  --output-root "$OUTPUT_ROOT" \
  --imgsz 640 --device 0 --batch 1 --warmup 20 --iterations 100
```

Validation saves P/R, mAP50, mAP75, mAP50:95, per-class AP, PR curves, normalized/raw confusion matrices, JSON, and CSV. Profiling must use the same GPU, image size, batch 1, warmup, and iteration count for B1–B5.

### 6.4 EarlyFusion-P2 decision experiment

`B3_early_fusion_p2.yaml` is a controlled follow-up, not a new core contribution. It isolates the P2 gain from LCMF. After switching to GPU mode, run a single-batch preflight first:

```bash
python scripts/train_b_baseline.py \
  --config configs/experiments/B3_early_fusion_p2.yaml \
  --data-root "$DATA_ROOT" --vis-dir Vis --ir-dir Ir --label-dir labels \
  --split-dir configs/splits/m3fd_seed42 --output-root "$OUTPUT_ROOT" \
  --device 0 --dry-run
```

Only after it passes should the same command be run without `--dry-run`. Compare B3, B3-P2, B4, and B5 under the same fixed split and protocol. If B3-P2 matches or exceeds B5 at materially lower inference cost, use EarlyFusion-P2 as the DRA baseline and remove LCMF from the final method.

## 7. Output safety

- Pair checks, statistics, and split generation refuse to overwrite files unless `--overwrite` is passed.
- Training always creates a unique run directory.
- Original M3FD images and labels are read-only inputs.
- Split files store sample IDs, not machine-specific absolute paths.

## 8. Current scope

Implemented:

- P0-ENV;
- P0-DATA;
- P0-FIXED-SPLIT;
- P0-STATS;
- P0-TINY-SMOKE;
- P0-SMALL-TRAIN;
- YOLOv8s-Visible 6.1 wrapper;
- paired B1–B5 baseline data loader;
- Visible, Infrared, EarlyFusion, LCMF, and LCMF-P2 model definitions;
- unified B1–B5 train/resume/validation/prediction/profile entry points.

Not implemented in this phase:

- DRA or validation-time DRA diagnostics;
- SR-style auxiliary;
- SegAux;
- ECA;
- SOTA comparisons;
- second datasets.

The B1–B5 code is prepared but has not yet completed GPU preflight or formal training in the current no-GPU instance.
