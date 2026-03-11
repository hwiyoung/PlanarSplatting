#!/bin/bash
# Phase 3-B: Ablation Study — 7 conditions, sequential GPU training
# Run inside Docker container: docker exec planarsplat bash scripts/run_phase3b_ablation.sh
set -e

cd /workspace/PlanarSplatting

OUT_BASE="planarSplat_ExpRes/phase3b"
RESULTS_BASE="results/phase3b"
DATA_PATH="user_inputs/testset/0_25x"

mkdir -p "${RESULTS_BASE}/eval" "${RESULTS_BASE}/images" "${RESULTS_BASE}/ply"

# Ablation conditions: letter, config_file, has_semantic
CONDITIONS=(
    "a:utils_demo/ablation_a_geo_only.conf:false"
    "b:utils_demo/ablation_b_sem_only.conf:true"
    "c:utils_demo/ablation_c_independent.conf:true"
    "d:utils_demo/ablation_d_joint.conf:true"
    "e:utils_demo/ablation_e_sem2geo.conf:true"
    "f:utils_demo/ablation_f_geo2sem.conf:true"
    "g:utils_demo/ablation_g_no_warmup.conf:true"
)

# Views for rendering (evenly spaced across 100 views)
VIEWS="0 12 25 37 50 62 75 87 99"

for COND in "${CONDITIONS[@]}"; do
    IFS=':' read -r LETTER CONF_FILE HAS_SEMANTIC <<< "$COND"

    echo ""
    echo "================================================================"
    echo "  ABLATION ($LETTER): $CONF_FILE"
    echo "  Started at: $(date)"
    echo "================================================================"

    # --- Training ---
    echo "[${LETTER}] Starting training..."
    python run_demo_colmap.py \
        -d "${DATA_PATH}" \
        -o "${OUT_BASE}" \
        --conf_path "${CONF_FILE}" \
        --use_precomputed_data

    # Find the latest checkpoint
    # Directory naming: {expname}_{scan_id}/{timestamp}/
    EXPNAME=$(grep "expname" "${CONF_FILE}" | head -1 | awk -F'=' '{print $2}' | tr -d ' ')
    SCAN_ID=$(grep "scan_id" "${CONF_FILE}" | head -1 | awk -F'=' '{print $2}' | tr -d ' ')
    LATEST_DIR=$(ls -dt "${OUT_BASE}/${EXPNAME}_${SCAN_ID}/"*/ 2>/dev/null | head -1)
    CKPT="${LATEST_DIR}checkpoints/Parameters/latest.pth"

    if [ ! -f "$CKPT" ]; then
        echo "[${LETTER}] ERROR: Checkpoint not found at ${CKPT}"
        continue
    fi
    echo "[${LETTER}] Checkpoint: ${CKPT}"

    # --- Evaluation ---
    echo "[${LETTER}] Running evaluation..."
    EVAL_METRICS="depth_mae normal_cos"
    if [ "$HAS_SEMANTIC" = "true" ]; then
        EVAL_METRICS="depth_mae normal_cos semantic_miou"
    fi
    python scripts/evaluate.py \
        --checkpoint "${CKPT}" \
        --metrics ${EVAL_METRICS} \
        --output "${RESULTS_BASE}/eval/abl_${LETTER}_${EXPNAME#abl_${LETTER}_}.json"

    # --- Render views ---
    echo "[${LETTER}] Rendering views..."
    python scripts/render_views.py \
        --checkpoint "${CKPT}" \
        --output_dir "${RESULTS_BASE}/images/${LETTER}" \
        --views ${VIEWS}

    # --- PLY export (normal + class) ---
    echo "[${LETTER}] Exporting PLY (normal)..."
    python scripts/visualize_primitives.py \
        --checkpoint "${CKPT}" \
        --color_by normal \
        --export_ply "${RESULTS_BASE}/ply/${LETTER}_normal.ply"

    if [ "$HAS_SEMANTIC" = "true" ]; then
        echo "[${LETTER}] Exporting PLY (class)..."
        python scripts/visualize_primitives.py \
            --checkpoint "${CKPT}" \
            --color_by class \
            --export_ply "${RESULTS_BASE}/ply/${LETTER}_class.ply"
    fi

    echo "[${LETTER}] Completed at: $(date)"
    echo ""
done

echo "================================================================"
echo "  All ablation trainings complete!"
echo "  Running comparison..."
echo "================================================================"

# --- Comparison table ---
python scripts/compare_ablation.py \
    --results_dir "${RESULTS_BASE}/eval" \
    --output_csv "${RESULTS_BASE}/ablation_comparison.csv"

echo ""
echo "Phase 3-B ablation study finished at: $(date)"
echo "Results: ${RESULTS_BASE}/"
echo "  - eval/*.json       : per-condition evaluation metrics"
echo "  - images/*/          : rendered depth/normal/semantic/RGB per view"
echo "  - ply/*_{normal,class}.ply : 3D primitive exports"
echo "  - ablation_comparison.csv  : comparison table"
