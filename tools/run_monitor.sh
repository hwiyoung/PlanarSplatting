#!/usr/bin/env bash
set -euo pipefail

TB_LOGDIR="${TB_LOGDIR:-/workspace/PlanarSplatting/planarSplat_ExpRes/expRes}"
TB_HOST="${TB_HOST:-0.0.0.0}"
TB_PORT="${TB_PORT:-6006}"
TB_PUBLIC_PORT="${TB_PUBLIC_PORT:-${TB_PORT}}"
INPUT_ROOT="${INPUT_ROOT:-/workspace/PlanarSplatting/user_inputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/PlanarSplatting/planarSplat_ExpRes}"
OUTPUT_HOST_PATH="${OUTPUT_HOST_PATH:-./planarSplat_ExpRes}"
TRAIN_CMD="${TRAIN_CMD:-}"
MAX_VGGT_IMAGES="${MAX_VGGT_IMAGES:-24}"
KEEP_ALIVE_AFTER_TRAIN="${KEEP_ALIVE_AFTER_TRAIN:-1}"
PLOT_FREQ="${PLOT_FREQ:-200}"
OVERWRITE_EXP="${OVERWRITE_EXP:-0}"
USE_PRECOMPUTED_DATA="${USE_PRECOMPUTED_DATA:-0}"
DATA_PATH=""
TRAIN_RC=0

if [[ -z "${TRAIN_CMD}" ]]; then
  if [[ -d "${INPUT_ROOT}/images" ]]; then
    DATA_PATH="${INPUT_ROOT}/images"
  else
    DATA_PATH="${INPUT_ROOT}"
  fi
  FRAME_STEP=1
  if [[ -d "${DATA_PATH}" ]]; then
    IMG_COUNT=$(find "${DATA_PATH}" -maxdepth 1 -type f | wc -l | tr -d ' ')
    if [[ "${IMG_COUNT}" -gt "${MAX_VGGT_IMAGES}" ]]; then
      FRAME_STEP=$(( (IMG_COUNT + MAX_VGGT_IMAGES - 1) / MAX_VGGT_IMAGES ))
    fi
  fi
  EXTRA_FLAGS="--plot_freq ${PLOT_FREQ}"
  if [[ "${OVERWRITE_EXP}" == "1" ]]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --overwrite_exp"
  fi
  if [[ "${USE_PRECOMPUTED_DATA}" == "1" ]]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --use_precomputed_data"
  fi
  TRAIN_CMD="python run_demo.py --data_path ${DATA_PATH} --frame_step ${FRAME_STEP} --out_path ${OUTPUT_ROOT}/demo ${EXTRA_FLAGS}"
fi

if [[ "${TRAIN_CMD}" == "python run_demo.py"* ]] && [[ -n "${DATA_PATH}" ]]; then
  if [[ ! -d "${DATA_PATH}" ]]; then
    echo "[monitor][ERR] Input path does not exist: ${DATA_PATH}"
    echo "[monitor][HINT] Set PLANARSPLAT_INPUT_DIR=/path/to/parent_or_images and rerun."
    exit 2
  fi
fi

echo "[monitor] TensorBoard logdir: ${TB_LOGDIR}"
echo "[monitor] TensorBoard URL:    http://localhost:${TB_PUBLIC_PORT}"
echo "[monitor] Input root:         ${INPUT_ROOT}"
echo "[monitor] Output root (ctr):  ${OUTPUT_ROOT}"
echo "[monitor] Output root (host): ${OUTPUT_HOST_PATH}"
if [[ -n "${DATA_PATH:-}" ]]; then
  echo "[monitor] Data path:          ${DATA_PATH}"
  if [[ -n "${IMG_COUNT:-}" ]]; then
    echo "[monitor] Image count:        ${IMG_COUNT}"
  fi
  if [[ -n "${FRAME_STEP:-}" ]]; then
    echo "[monitor] Auto frame_step:    ${FRAME_STEP} (MAX_VGGT_IMAGES=${MAX_VGGT_IMAGES})"
  fi
  echo "[monitor] Plot freq:          ${PLOT_FREQ}"
  echo "[monitor] Overwrite exp:      ${OVERWRITE_EXP}"
  echo "[monitor] Use precomputed:    ${USE_PRECOMPUTED_DATA}"
fi
echo "[monitor] Train command:      ${TRAIN_CMD}"

tensorboard --logdir "${TB_LOGDIR}" --host "${TB_HOST}" --port "${TB_PORT}" &
TB_PID=$!

cleanup() {
  if kill -0 "${TB_PID}" 2>/dev/null; then
    kill "${TB_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

set +e
bash -lc "${TRAIN_CMD}"
TRAIN_RC=$?
set -e

if [[ "${TRAIN_RC}" -ne 0 ]]; then
  echo "[monitor][WARN] Train command exited with code ${TRAIN_RC}."
else
  echo "[monitor][OK] Train command finished."
fi

if [[ "${KEEP_ALIVE_AFTER_TRAIN}" == "1" ]]; then
  echo "[monitor] Keeping TensorBoard alive for result tracking."
  echo "[monitor] Press Ctrl+C to stop."
  wait "${TB_PID}"
else
  exit "${TRAIN_RC}"
fi
