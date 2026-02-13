#!/usr/bin/env bash
set -euo pipefail

TB_LOGDIR="${TB_LOGDIR:-/workspace/PlanarSplatting/planarSplat_ExpRes/expRes}"
TB_HOST="${TB_HOST:-0.0.0.0}"
TB_PORT="${TB_PORT:-6006}"
TB_PUBLIC_PORT="${TB_PUBLIC_PORT:-${TB_PORT}}"
SYNC_VIEWER_ENABLE="${SYNC_VIEWER_ENABLE:-1}"
SYNC_VIEWER_HOST="${SYNC_VIEWER_HOST:-0.0.0.0}"
SYNC_VIEWER_PORT="${SYNC_VIEWER_PORT:-18080}"
SYNC_VIEWER_PUBLIC_PORT="${SYNC_VIEWER_PUBLIC_PORT:-${SYNC_VIEWER_PORT}}"
INPUT_ROOT="${INPUT_ROOT:-/workspace/PlanarSplatting/user_inputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/PlanarSplatting/planarSplat_ExpRes}"
OUTPUT_HOST_PATH="${OUTPUT_HOST_PATH:-./planarSplat_ExpRes}"
SYNC_VIEWER_OUTPUT_ROOT="${SYNC_VIEWER_OUTPUT_ROOT:-${OUTPUT_ROOT}/demo}"
SYNC_VIEWER_RUN_DIR="${SYNC_VIEWER_RUN_DIR:-}"
SYNC_VIEWER_RETRY_SEC="${SYNC_VIEWER_RETRY_SEC:-10}"
SYNC_VIEWER_REFRESH_SEC="${SYNC_VIEWER_REFRESH_SEC:-0}"
SYNC_VIEWER_POLL_SEC="${SYNC_VIEWER_POLL_SEC:-0}"
SYNC_VIEWER_COMPARE_RUNS="${SYNC_VIEWER_COMPARE_RUNS:-4}"
SYNC_VIEWER_POINT_STRIDE="${SYNC_VIEWER_POINT_STRIDE:-8}"
SYNC_VIEWER_MAX_POINTS="${SYNC_VIEWER_MAX_POINTS:-350000}"
SYNC_VIEWER_MAX_FACES="${SYNC_VIEWER_MAX_FACES:-500000}"
SYNC_VIEWER_MAX_CAMERAS="${SYNC_VIEWER_MAX_CAMERAS:-96}"
SYNC_VIEWER_FRUSTUM_RATIO="${SYNC_VIEWER_FRUSTUM_RATIO:-0.04}"
SYNC_VIEWER_PREFER_SAVED_PCD="${SYNC_VIEWER_PREFER_SAVED_PCD:-0}"
SYNC_VIEWER_HTML_PATH="${SYNC_VIEWER_HTML_PATH:-${OUTPUT_ROOT}/sync_viewer_live/sync_dual_viewer.html}"
SYNC_VIEWER_HTML_NAME="$(basename "${SYNC_VIEWER_HTML_PATH}")"
TRAIN_CMD="${TRAIN_CMD:-}"
MAX_VGGT_IMAGES="${MAX_VGGT_IMAGES:-24}"
KEEP_ALIVE_AFTER_TRAIN="${KEEP_ALIVE_AFTER_TRAIN:-1}"
PLOT_FREQ="${PLOT_FREQ:-200}"
OVERWRITE_EXP="${OVERWRITE_EXP:-0}"
USE_PRECOMPUTED_DATA="${USE_PRECOMPUTED_DATA:-0}"
DEPTH_CONF="${DEPTH_CONF:-1.0}"
DATA_PATH=""
TRAIN_RC=0
TB_PID=0
VIEWER_LOOP_PID=0
VIEWER_REFRESH_PID=0
SYNC_VIEWER_PCD_ARG=""
TRAIN_MODE="custom"
AUTO_TRAIN_ARGS=()
TRAIN_CMD_PRINT=""

if ! [[ "${SYNC_VIEWER_REFRESH_SEC}" =~ ^[0-9]+$ ]]; then
  echo "[planarsplat][WARN] Invalid SYNC_VIEWER_REFRESH_SEC='${SYNC_VIEWER_REFRESH_SEC}', forcing 0."
  SYNC_VIEWER_REFRESH_SEC=0
fi
if ! [[ "${SYNC_VIEWER_POLL_SEC}" =~ ^([0-9]+([.][0-9]+)?)$ ]]; then
  echo "[planarsplat][WARN] Invalid SYNC_VIEWER_POLL_SEC='${SYNC_VIEWER_POLL_SEC}', forcing 0."
  SYNC_VIEWER_POLL_SEC=0
fi
if ! [[ "${DEPTH_CONF}" =~ ^([0-9]+([.][0-9]+)?)$ ]]; then
  echo "[planarsplat][WARN] Invalid DEPTH_CONF='${DEPTH_CONF}', forcing 1.0."
  DEPTH_CONF=1.0
fi
if ! [[ "${SYNC_VIEWER_COMPARE_RUNS}" =~ ^[0-9]+$ ]] || [[ "${SYNC_VIEWER_COMPARE_RUNS}" -lt 1 ]]; then
  echo "[planarsplat][WARN] Invalid SYNC_VIEWER_COMPARE_RUNS='${SYNC_VIEWER_COMPARE_RUNS}', forcing 4."
  SYNC_VIEWER_COMPARE_RUNS=4
fi
if ! [[ "${SYNC_VIEWER_POINT_STRIDE}" =~ ^[0-9]+$ ]] || [[ "${SYNC_VIEWER_POINT_STRIDE}" -lt 1 ]]; then
  echo "[planarsplat][WARN] Invalid SYNC_VIEWER_POINT_STRIDE='${SYNC_VIEWER_POINT_STRIDE}', forcing 8."
  SYNC_VIEWER_POINT_STRIDE=8
fi
if ! [[ "${SYNC_VIEWER_MAX_POINTS}" =~ ^[0-9]+$ ]] || [[ "${SYNC_VIEWER_MAX_POINTS}" -lt 10000 ]]; then
  echo "[planarsplat][WARN] Invalid SYNC_VIEWER_MAX_POINTS='${SYNC_VIEWER_MAX_POINTS}', forcing 350000."
  SYNC_VIEWER_MAX_POINTS=350000
fi
if ! [[ "${SYNC_VIEWER_MAX_FACES}" =~ ^[0-9]+$ ]] || [[ "${SYNC_VIEWER_MAX_FACES}" -lt 10000 ]]; then
  echo "[planarsplat][WARN] Invalid SYNC_VIEWER_MAX_FACES='${SYNC_VIEWER_MAX_FACES}', forcing 500000."
  SYNC_VIEWER_MAX_FACES=500000
fi
if ! [[ "${SYNC_VIEWER_MAX_CAMERAS}" =~ ^[0-9]+$ ]] || [[ "${SYNC_VIEWER_MAX_CAMERAS}" -lt 1 ]]; then
  echo "[planarsplat][WARN] Invalid SYNC_VIEWER_MAX_CAMERAS='${SYNC_VIEWER_MAX_CAMERAS}', forcing 96."
  SYNC_VIEWER_MAX_CAMERAS=96
fi
if ! [[ "${SYNC_VIEWER_FRUSTUM_RATIO}" =~ ^([0-9]+([.][0-9]+)?)$ ]]; then
  echo "[planarsplat][WARN] Invalid SYNC_VIEWER_FRUSTUM_RATIO='${SYNC_VIEWER_FRUSTUM_RATIO}', forcing 0.04."
  SYNC_VIEWER_FRUSTUM_RATIO=0.04
fi
if ! [[ "${SYNC_VIEWER_PREFER_SAVED_PCD}" =~ ^[01]$ ]]; then
  echo "[planarsplat][WARN] Invalid SYNC_VIEWER_PREFER_SAVED_PCD='${SYNC_VIEWER_PREFER_SAVED_PCD}', forcing 0."
  SYNC_VIEWER_PREFER_SAVED_PCD=0
fi
if [[ "${SYNC_VIEWER_PREFER_SAVED_PCD}" == "1" ]]; then
  SYNC_VIEWER_PCD_ARG="--prefer_saved_pcd"
fi

if [[ -z "${TRAIN_CMD}" ]]; then
  if [[ -d "${INPUT_ROOT}/images" ]]; then
    DATA_PATH="${INPUT_ROOT}/images"
  else
    DATA_PATH="${INPUT_ROOT}"
  fi
  FRAME_STEP=1
  if [[ -d "${DATA_PATH}" ]]; then
    IMG_COUNT=$(find "${DATA_PATH}" -maxdepth 1 -type f \( \
      -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o \
      -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' -o \
      -iname '*.webp' \
    \) | wc -l | tr -d ' ')
    if [[ "${IMG_COUNT}" -gt "${MAX_VGGT_IMAGES}" ]]; then
      FRAME_STEP=$(( (IMG_COUNT + MAX_VGGT_IMAGES - 1) / MAX_VGGT_IMAGES ))
    fi
  fi
  AUTO_TRAIN_ARGS=(
    python run_demo.py
    --data_path "${DATA_PATH}"
    --frame_step "${FRAME_STEP}"
    --depth_conf "${DEPTH_CONF}"
    --out_path "${OUTPUT_ROOT}/demo"
    --plot_freq "${PLOT_FREQ}"
  )
  if [[ "${OVERWRITE_EXP}" == "1" ]]; then
    AUTO_TRAIN_ARGS+=(--overwrite_exp)
  fi
  if [[ "${USE_PRECOMPUTED_DATA}" == "1" ]]; then
    AUTO_TRAIN_ARGS+=(--use_precomputed_data)
  fi
  TRAIN_MODE="auto"
  TRAIN_CMD_PRINT="$(printf '%q ' "${AUTO_TRAIN_ARGS[@]}")"
else
  TRAIN_MODE="custom"
  TRAIN_CMD_PRINT="${TRAIN_CMD}"
fi

if [[ "${TRAIN_MODE}" == "auto" ]] && [[ -n "${DATA_PATH}" ]]; then
  if [[ ! -d "${DATA_PATH}" ]]; then
    echo "[planarsplat][ERR] Input path does not exist: ${DATA_PATH}"
    echo "[planarsplat][HINT] Set PLANARSPLAT_INPUT_DIR=/path/to/parent_or_images and rerun."
    exit 2
  fi
fi

echo "[planarsplat] TensorBoard logdir: ${TB_LOGDIR}"
echo "[planarsplat] TensorBoard URL:    http://localhost:${TB_PUBLIC_PORT}"
if [[ "${SYNC_VIEWER_ENABLE}" == "1" ]]; then
  echo "[planarsplat] Sync viewer URL:   http://localhost:${SYNC_VIEWER_PUBLIC_PORT}/${SYNC_VIEWER_HTML_NAME}"
  if [[ "${SYNC_VIEWER_REFRESH_SEC}" -gt 0 ]]; then
    echo "[planarsplat] Sync viewer live:  refresh=${SYNC_VIEWER_REFRESH_SEC}s, poll=${SYNC_VIEWER_POLL_SEC}s"
  else
    echo "[planarsplat] Sync viewer live:  disabled (performance-safe default)"
  fi
  echo "[planarsplat] Sync viewer runs:  compare_runs=${SYNC_VIEWER_COMPARE_RUNS}"
  echo "[planarsplat] Sync viewer geom:  stride=${SYNC_VIEWER_POINT_STRIDE}, points=${SYNC_VIEWER_MAX_POINTS}, faces=${SYNC_VIEWER_MAX_FACES}"
fi
echo "[planarsplat] Input root:         ${INPUT_ROOT}"
echo "[planarsplat] Output root (ctr):  ${OUTPUT_ROOT}"
echo "[planarsplat] Output root (host): ${OUTPUT_HOST_PATH}"
if [[ -n "${DATA_PATH:-}" ]]; then
  echo "[planarsplat] Data path:          ${DATA_PATH}"
  if [[ -n "${IMG_COUNT:-}" ]]; then
    echo "[planarsplat] Image count:        ${IMG_COUNT}"
  fi
  if [[ -n "${FRAME_STEP:-}" ]]; then
    echo "[planarsplat] Auto frame_step:    ${FRAME_STEP} (MAX_VGGT_IMAGES=${MAX_VGGT_IMAGES})"
  fi
  echo "[planarsplat] Plot freq:          ${PLOT_FREQ}"
  echo "[planarsplat] Depth conf:         ${DEPTH_CONF}"
  echo "[planarsplat] Overwrite exp:      ${OVERWRITE_EXP}"
  echo "[planarsplat] Use precomputed:    ${USE_PRECOMPUTED_DATA}"
fi
echo "[planarsplat] Train mode:         ${TRAIN_MODE}"
echo "[planarsplat] Train command:      ${TRAIN_CMD_PRINT}"
if [[ "${TRAIN_MODE}" == "custom" ]]; then
  echo "[planarsplat][WARN] Running custom TRAIN_CMD via shell."
fi

start_sync_viewer_loop() {
  if [[ "${SYNC_VIEWER_ENABLE}" != "1" ]]; then
    echo "[planarsplat] Sync viewer disabled (SYNC_VIEWER_ENABLE=${SYNC_VIEWER_ENABLE})."
    return
  fi
  (
    while true; do
      RC=0
      if [[ -n "${SYNC_VIEWER_RUN_DIR}" ]]; then
        if [[ -d "${SYNC_VIEWER_RUN_DIR}" ]]; then
          if python tools/run_sync_dual_viewer.py \
            --run_dir "${SYNC_VIEWER_RUN_DIR}" \
            --html_path "${SYNC_VIEWER_HTML_PATH}" \
            --compare_runs "${SYNC_VIEWER_COMPARE_RUNS}" \
            --point_stride "${SYNC_VIEWER_POINT_STRIDE}" \
            --max_points "${SYNC_VIEWER_MAX_POINTS}" \
            --max_faces "${SYNC_VIEWER_MAX_FACES}" \
            --max_cameras "${SYNC_VIEWER_MAX_CAMERAS}" \
            --frustum_ratio "${SYNC_VIEWER_FRUSTUM_RATIO}" \
            ${SYNC_VIEWER_PCD_ARG} \
            --live_poll_sec "${SYNC_VIEWER_POLL_SEC}" \
            --serve --host "${SYNC_VIEWER_HOST}" --port "${SYNC_VIEWER_PORT}"; then
            RC=0
          else
            RC=$?
          fi
        else
          RC=1
        fi
      else
        if python tools/run_sync_dual_viewer.py \
          --output_root "${SYNC_VIEWER_OUTPUT_ROOT}" \
          --html_path "${SYNC_VIEWER_HTML_PATH}" \
          --compare_runs "${SYNC_VIEWER_COMPARE_RUNS}" \
          --point_stride "${SYNC_VIEWER_POINT_STRIDE}" \
          --max_points "${SYNC_VIEWER_MAX_POINTS}" \
          --max_faces "${SYNC_VIEWER_MAX_FACES}" \
          --max_cameras "${SYNC_VIEWER_MAX_CAMERAS}" \
          --frustum_ratio "${SYNC_VIEWER_FRUSTUM_RATIO}" \
          ${SYNC_VIEWER_PCD_ARG} \
          --live_poll_sec "${SYNC_VIEWER_POLL_SEC}" \
          --serve --host "${SYNC_VIEWER_HOST}" --port "${SYNC_VIEWER_PORT}"; then
          RC=0
        else
          RC=$?
        fi
      fi

      if [[ "${RC}" -eq 0 ]]; then
        exit 0
      fi
      if [[ "${RC}" -eq 3 ]]; then
        echo "[planarsplat][WARN] Sync viewer disabled due to missing open3d (rc=${RC})."
        exit 0
      fi
      sleep "${SYNC_VIEWER_RETRY_SEC}"
    done
  ) &
  VIEWER_LOOP_PID=$!
}

start_sync_viewer_refresh_loop() {
  if [[ "${SYNC_VIEWER_ENABLE}" != "1" ]]; then
    return
  fi
  if [[ "${SYNC_VIEWER_REFRESH_SEC}" -le 0 ]]; then
    return
  fi
  (
    sleep "${SYNC_VIEWER_REFRESH_SEC}"
    while true; do
      RC=0
      if [[ -n "${SYNC_VIEWER_RUN_DIR}" ]]; then
        if [[ -d "${SYNC_VIEWER_RUN_DIR}" ]]; then
          if python tools/run_sync_dual_viewer.py \
            --run_dir "${SYNC_VIEWER_RUN_DIR}" \
            --html_path "${SYNC_VIEWER_HTML_PATH}" \
            --compare_runs "${SYNC_VIEWER_COMPARE_RUNS}" \
            --point_stride "${SYNC_VIEWER_POINT_STRIDE}" \
            --max_points "${SYNC_VIEWER_MAX_POINTS}" \
            --max_faces "${SYNC_VIEWER_MAX_FACES}" \
            --max_cameras "${SYNC_VIEWER_MAX_CAMERAS}" \
            --frustum_ratio "${SYNC_VIEWER_FRUSTUM_RATIO}" \
            ${SYNC_VIEWER_PCD_ARG} \
            --live_poll_sec "${SYNC_VIEWER_POLL_SEC}"; then
            RC=0
          else
            RC=$?
          fi
        else
          RC=1
        fi
      else
        if python tools/run_sync_dual_viewer.py \
          --output_root "${SYNC_VIEWER_OUTPUT_ROOT}" \
          --html_path "${SYNC_VIEWER_HTML_PATH}" \
          --compare_runs "${SYNC_VIEWER_COMPARE_RUNS}" \
          --point_stride "${SYNC_VIEWER_POINT_STRIDE}" \
          --max_points "${SYNC_VIEWER_MAX_POINTS}" \
          --max_faces "${SYNC_VIEWER_MAX_FACES}" \
          --max_cameras "${SYNC_VIEWER_MAX_CAMERAS}" \
          --frustum_ratio "${SYNC_VIEWER_FRUSTUM_RATIO}" \
          ${SYNC_VIEWER_PCD_ARG} \
          --live_poll_sec "${SYNC_VIEWER_POLL_SEC}"; then
          RC=0
        else
          RC=$?
        fi
      fi

      if [[ "${RC}" -eq 3 ]]; then
        echo "[planarsplat][WARN] Sync viewer refresh disabled due to missing open3d (rc=${RC})."
        exit 0
      fi
      sleep "${SYNC_VIEWER_REFRESH_SEC}"
    done
  ) &
  VIEWER_REFRESH_PID=$!
}

refresh_sync_viewer_once() {
  if [[ "${SYNC_VIEWER_ENABLE}" != "1" ]]; then
    return
  fi
  set +e
  if [[ -n "${SYNC_VIEWER_RUN_DIR}" ]]; then
    if [[ -d "${SYNC_VIEWER_RUN_DIR}" ]]; then
      python tools/run_sync_dual_viewer.py \
        --run_dir "${SYNC_VIEWER_RUN_DIR}" \
        --html_path "${SYNC_VIEWER_HTML_PATH}" \
        --compare_runs "${SYNC_VIEWER_COMPARE_RUNS}" \
        --point_stride "${SYNC_VIEWER_POINT_STRIDE}" \
        --max_points "${SYNC_VIEWER_MAX_POINTS}" \
        --max_faces "${SYNC_VIEWER_MAX_FACES}" \
        --max_cameras "${SYNC_VIEWER_MAX_CAMERAS}" \
        --frustum_ratio "${SYNC_VIEWER_FRUSTUM_RATIO}" \
        ${SYNC_VIEWER_PCD_ARG} \
        --live_poll_sec "${SYNC_VIEWER_POLL_SEC}"
      RC=$?
    else
      RC=1
    fi
  else
    python tools/run_sync_dual_viewer.py \
      --output_root "${SYNC_VIEWER_OUTPUT_ROOT}" \
      --html_path "${SYNC_VIEWER_HTML_PATH}" \
      --compare_runs "${SYNC_VIEWER_COMPARE_RUNS}" \
      --point_stride "${SYNC_VIEWER_POINT_STRIDE}" \
      --max_points "${SYNC_VIEWER_MAX_POINTS}" \
      --max_faces "${SYNC_VIEWER_MAX_FACES}" \
      --max_cameras "${SYNC_VIEWER_MAX_CAMERAS}" \
      --frustum_ratio "${SYNC_VIEWER_FRUSTUM_RATIO}" \
      ${SYNC_VIEWER_PCD_ARG} \
      --live_poll_sec "${SYNC_VIEWER_POLL_SEC}"
    RC=$?
  fi
  set -e
  if [[ "${RC}" -ne 0 ]]; then
    echo "[planarsplat][WARN] One-shot sync viewer refresh failed (rc=${RC})."
  else
    echo "[planarsplat] Sync viewer refreshed with latest run artifacts."
  fi
}

run_custom_train_cmd() {
  local cmd="$1"
  if [[ -z "${cmd}" ]]; then
    echo "[planarsplat][ERR] TRAIN_CMD is empty."
    return 2
  fi
  case "${cmd}" in
    *[\|\&\;\<\>\(\)\`\$]*)
      echo "[planarsplat][WARN] TRAIN_CMD contains shell operators; executing via shell for compatibility."
      bash -lc "${cmd}"
      return $?
      ;;
    *)
      python - "${cmd}" <<'PY'
import os
import re
import shlex
import subprocess
import sys

cmd = sys.argv[1]
args = shlex.split(cmd)
if not args:
    print("[planarsplat][ERR] TRAIN_CMD split to empty argv.")
    raise SystemExit(2)
env = os.environ.copy()
var_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
idx = 0
while idx < len(args):
    token = args[idx]
    if token == "--":
        idx += 1
        break
    if "=" not in token or token.startswith("="):
        break
    key, value = token.split("=", 1)
    if not var_re.match(key):
        break
    env[key] = value
    idx += 1

argv = args[idx:]
if not argv:
    print("[planarsplat][ERR] TRAIN_CMD contains only env assignments; missing executable.")
    raise SystemExit(2)

raise SystemExit(subprocess.call(argv, env=env))
PY
      return $?
      ;;
  esac
}

tensorboard --logdir "${TB_LOGDIR}" --host "${TB_HOST}" --port "${TB_PORT}" &
TB_PID=$!
start_sync_viewer_loop
start_sync_viewer_refresh_loop

cleanup() {
  if [[ "${TB_PID}" -ne 0 ]] && kill -0 "${TB_PID}" 2>/dev/null; then
    kill "${TB_PID}" 2>/dev/null || true
  fi
  if [[ "${VIEWER_LOOP_PID}" -ne 0 ]] && kill -0 "${VIEWER_LOOP_PID}" 2>/dev/null; then
    kill "${VIEWER_LOOP_PID}" 2>/dev/null || true
  fi
  if [[ "${VIEWER_REFRESH_PID}" -ne 0 ]] && kill -0 "${VIEWER_REFRESH_PID}" 2>/dev/null; then
    kill "${VIEWER_REFRESH_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

set +e
if [[ "${TRAIN_MODE}" == "auto" ]]; then
  "${AUTO_TRAIN_ARGS[@]}"
else
  run_custom_train_cmd "${TRAIN_CMD}"
fi
TRAIN_RC=$?
set -e

if [[ "${TRAIN_RC}" -ne 0 ]]; then
  echo "[planarsplat][WARN] Train command exited with code ${TRAIN_RC}."
else
  echo "[planarsplat][OK] Train command finished."
fi

refresh_sync_viewer_once

if [[ "${KEEP_ALIVE_AFTER_TRAIN}" == "1" ]]; then
  echo "[planarsplat] Keeping TensorBoard alive for result tracking."
  echo "[planarsplat] Press Ctrl+C to stop."
  wait "${TB_PID}"
else
  exit "${TRAIN_RC}"
fi
