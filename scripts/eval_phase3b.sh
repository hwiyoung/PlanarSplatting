#!/bin/bash
# Phase 3-B: Post-hoc evaluation for completed ablation conditions
# Run inside Docker container: docker exec planarsplat bash scripts/eval_phase3b.sh
set -e

cd /workspace/PlanarSplatting

OUT_BASE="planarSplat_ExpRes/phase3b"
RESULTS_BASE="results/phase3b"

mkdir -p "${RESULTS_BASE}/eval" "${RESULTS_BASE}/images" "${RESULTS_BASE}/ply"

# Views for rendering (evenly spaced across 100 views)
VIEWS="0 12 25 37 50 62 75 87 99"

# Ablation conditions: letter, has_semantic
CONDITIONS=(
    "a:abl_a_geo_only_example:false"
    "b:abl_b_sem_only_example:true"
    "c:abl_c_independent_example:true"
    "d:abl_d_joint_example:true"
    "e:abl_e_sem2geo_example:true"
    "f:abl_f_geo2sem_example:true"
    "g:abl_g_no_warmup_example:true"
)

for COND in "${CONDITIONS[@]}"; do
    IFS=':' read -r LETTER EXPDIR HAS_SEMANTIC <<< "$COND"

    # Find latest checkpoint
    LATEST_DIR=$(ls -dt "${OUT_BASE}/${EXPDIR}/"*/ 2>/dev/null | head -1)
    CKPT="${LATEST_DIR}checkpoints/Parameters/latest.pth"

    if [ ! -f "$CKPT" ]; then
        echo "[${LETTER}] Checkpoint not found, skipping (${CKPT})"
        continue
    fi
    echo ""
    echo "================================================================"
    echo "  EVALUATING ($LETTER): ${CKPT}"
    echo "================================================================"

    # --- Evaluation ---
    EVAL_METRICS="depth_mae normal_cos"
    if [ "$HAS_SEMANTIC" = "true" ]; then
        EVAL_METRICS="depth_mae normal_cos semantic_miou"
    fi

    echo "[${LETTER}] Running evaluation..."
    python scripts/evaluate.py \
        --checkpoint "${CKPT}" \
        --metrics ${EVAL_METRICS} \
        --output "${RESULTS_BASE}/eval/abl_${LETTER}.json"

    # --- Render views ---
    echo "[${LETTER}] Rendering views..."
    python scripts/render_views.py \
        --checkpoint "${CKPT}" \
        --output_dir "${RESULTS_BASE}/images/${LETTER}" \
        --views ${VIEWS}

    # --- PLY export ---
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

    echo "[${LETTER}] Done!"
done

# --- Comparison table ---
echo ""
echo "Running comparison..."
python scripts/compare_ablation.py \
    --results_dir "${RESULTS_BASE}/eval" \
    --output_csv "${RESULTS_BASE}/ablation_comparison.csv"

echo ""
echo "Evaluation complete! Results: ${RESULTS_BASE}/"
