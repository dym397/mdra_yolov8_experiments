#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DATA_ROOT=""
OUTPUT_ROOT="$PROJECT_ROOT/outputs"
SPLIT_DIR="$PROJECT_ROOT/configs/splits/m3fd_seed42"
VIS_DIR="Vis"
IR_DIR="Ir"
LABEL_DIR="labels"
DEVICE="0"
SKIP_PREFLIGHT=0
ONLY_CONFIG=""

usage() {
  cat <<'EOF'
Usage: bash scripts/run_b_suite.sh --data-root PATH [options]

Options:
  --output-root PATH       Default: <project>/outputs
  --split-dir PATH         Default: configs/splits/m3fd_seed42
  --vis-dir NAME           Default: Vis
  --ir-dir NAME            Default: Ir
  --label-dir NAME         Default: labels
  --device VALUE           Default: 0
  --only ID                Run only configs/experiments/ID.yaml
  --skip-preflight         Use only after all five GPU preflights passed
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --split-dir) SPLIT_DIR="$2"; shift 2 ;;
    --vis-dir) VIS_DIR="$2"; shift 2 ;;
    --ir-dir) IR_DIR="$2"; shift 2 ;;
    --label-dir) LABEL_DIR="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --only) ONLY_CONFIG="$2"; shift 2 ;;
    --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$DATA_ROOT" ]] || { echo "--data-root is required" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 2; }
[[ -d "$DATA_ROOT" ]] || { echo "Dataset root does not exist: $DATA_ROOT" >&2; exit 2; }
[[ -d "$SPLIT_DIR" ]] || { echo "Split directory does not exist: $SPLIT_DIR" >&2; exit 2; }

mkdir -p "$OUTPUT_ROOT/b_suite_controller"
exec 9>"$OUTPUT_ROOT/b_suite_controller.lock"
flock -n 9 || { echo "Another B-suite controller is already running" >&2; exit 3; }

RUN_ID="b_suite_$(date '+%Y%m%d_%H%M%S')"
CONTROLLER_DIR="$OUTPUT_ROOT/b_suite_controller/$RUN_ID"
mkdir -p "$CONTROLLER_DIR/logs"
printf '%s\n' "$CONTROLLER_DIR" > "$OUTPUT_ROOT/b_suite_controller/latest.txt"
STATUS_FILE="$CONTROLLER_DIR/status.txt"
HISTORY_FILE="$CONTROLLER_DIR/history.tsv"
CURRENT_STEP="initializing"
export PYTHONUNBUFFERED=1

write_status() {
  local state="$1" step="$2" message="$3" now tmp
  now="$(date --iso-8601=seconds)"
  tmp="$STATUS_FILE.tmp"
  {
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'pid=%s\n' "$$"
    printf 'state=%s\n' "$state"
    printf 'step=%s\n' "$step"
    printf 'updated_at=%s\n' "$now"
    printf 'message=%s\n' "$message"
  } > "$tmp"
  mv "$tmp" "$STATUS_FILE"
  printf '%s\t%s\t%s\t%s\n' "$now" "$state" "$step" "$message" >> "$HISTORY_FILE"
}

on_error() {
  local rc=$?
  write_status "failed" "$CURRENT_STEP" "controller stopped with exit code $rc"
  exit "$rc"
}
trap on_error ERR INT TERM

run_step() {
  local step="$1"; shift
  local log="$CONTROLLER_DIR/logs/${step}.log"
  CURRENT_STEP="$step"
  write_status "running" "$step" "log=$log"
  set +e
  "$@" > "$log" 2>&1
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    write_status "failed" "$step" "exit_code=$rc log=$log"
    return "$rc"
  fi
  write_status "passed" "$step" "log=$log"
}

CONFIGS=(B1_visible B2_infrared B3_early_fusion B4_lcmf B5_lcmf_p2)
if [[ -n "$ONLY_CONFIG" ]]; then
  [[ -f "$PROJECT_ROOT/configs/experiments/${ONLY_CONFIG}.yaml" ]] || {
    echo "Experiment config does not exist: configs/experiments/${ONLY_CONFIG}.yaml" >&2
    exit 2
  }
  CONFIGS=("$ONLY_CONFIG")
fi
COMMON_ARGS=(
  --data-root "$DATA_ROOT"
  --vis-dir "$VIS_DIR"
  --ir-dir "$IR_DIR"
  --label-dir "$LABEL_DIR"
  --split-dir "$SPLIT_DIR"
  --output-root "$OUTPUT_ROOT"
  --device "$DEVICE"
)

cd "$PROJECT_ROOT"
write_status "running" "initializing" "controller_dir=$CONTROLLER_DIR"

if [[ $SKIP_PREFLIGHT -eq 0 ]]; then
  for id in "${CONFIGS[@]}"; do
    run_step "${id}_preflight" \
      "$PYTHON_BIN" scripts/train_b_baseline.py \
      --config "configs/experiments/${id}.yaml" "${COMMON_ARGS[@]}" --dry-run
  done
fi

for id in "${CONFIGS[@]}"; do
  run_step "${id}_train" \
    "$PYTHON_BIN" scripts/train_b_baseline.py \
    --config "configs/experiments/${id}.yaml" "${COMMON_ARGS[@]}"

  run_dir="$(find "$OUTPUT_ROOT/b_experiments" -mindepth 1 -maxdepth 1 -type d -name "${id}*" \
    -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
  [[ -n "$run_dir" && -f "$run_dir/weights/best.pt" ]] || {
    echo "Cannot locate best checkpoint for $id" >&2
    false
  }
  printf '%s\n' "$run_dir" > "$CONTROLLER_DIR/${id}_run_dir.txt"

  checkpoint="$run_dir/weights/best.pt"
  if [[ -f "$run_dir/weights/best_inference.pt" ]]; then
    checkpoint="$run_dir/weights/best_inference.pt"
  fi

  run_step "${id}_test" \
    "$PYTHON_BIN" scripts/validate_b_baseline.py \
    --checkpoint "$checkpoint" --split test \
    --output-root "$OUTPUT_ROOT" --device "$DEVICE" --workers 4

  run_step "${id}_predict" \
    "$PYTHON_BIN" scripts/predict_b_baseline.py \
    --checkpoint "$checkpoint" --data-root "$DATA_ROOT" --split-dir "$SPLIT_DIR" \
    --split test --max-images 20 --output-root "$OUTPUT_ROOT" --device "$DEVICE"

  run_step "${id}_profile" \
    "$PYTHON_BIN" scripts/profile_b_baseline.py \
    --checkpoint "$checkpoint" --output-root "$OUTPUT_ROOT" \
    --imgsz 640 --device "$DEVICE" --batch 1 --warmup 20 --iterations 100

  write_status "model_completed" "$id" "run_dir=$run_dir"
done

CURRENT_STEP="completed"
write_status "completed" "completed" "all selected experiments and post-processing passed"
