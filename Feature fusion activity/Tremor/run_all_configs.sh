#!/usr/bin/env bash
# run_all_configs.sh
#
# Runs the full multi-seed ablation study for all 5 dataset configurations
# sequentially. Each config runs 11 seeds (10–20) across all 8 k-values.
#
# Usage:
#   bash run_all_configs.sh
#   bash run_all_configs.sh 2>&1 | tee run_all_configs.log   # also save log
#
# Estimated time: ~5 × (time for one multirun)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$(which python)"
MULTIRUN="$SCRIPT_DIR/run_ablation_multirun_aulus5.py"

SEEDS="10,11,12,13,14,15,16,17,18,19,20"
GPUS="0,1"

echo "========================================================"
echo "  Tremor Ablation — All Configurations"
echo "  Seeds   : $SEEDS"
echo "  GPUs    : $GPUS"
echo "  Started : $(date)"
echo "========================================================"

# ── 1. clean ─────────────────────────────────────────────────────────────────
echo ""
echo "### Config 1/5: clean ###"
"$PYTHON" "$MULTIRUN" \
    --seeds "$SEEDS" \
    --gpus  "$GPUS" \
    --tag   "clean" \
    --variants "s4_w4_fs50_tremor_clean"

# ── 2. clean_aug ─────────────────────────────────────────────────────────────
echo ""
echo "### Config 2/5: clean_aug ###"
"$PYTHON" "$MULTIRUN" \
    --seeds "$SEEDS" \
    --gpus  "$GPUS" \
    --tag   "clean_aug" \
    --variants "s4_w4_fs50_tremor_clean,s4_w4_fs50_tremor_clean_awgn,s4_w4_fs50_tremor_clean_rotation"

# ── 3. mixed ─────────────────────────────────────────────────────────────────
echo ""
echo "### Config 3/5: mixed ###"
"$PYTHON" "$MULTIRUN" \
    --seeds "$SEEDS" \
    --gpus  "$GPUS" \
    --tag   "mixed" \
    --variants "s4_w4_fs50_tremor_clean,s2_w2_fs50_tremor_mild_mod,s2_w2_fs50_tremor_mod_severe"

# ── 4. tremor ────────────────────────────────────────────────────────────────
echo ""
echo "### Config 4/5: tremor ###"
"$PYTHON" "$MULTIRUN" \
    --seeds "$SEEDS" \
    --gpus  "$GPUS" \
    --tag   "tremor" \
    --variants "s2_w2_fs50_tremor_mild_mod,s2_w2_fs50_tremor_mod_severe"

# ── 5. tremor_severe ─────────────────────────────────────────────────────────
echo ""
echo "### Config 5/5: tremor_severe ###"
"$PYTHON" "$MULTIRUN" \
    --seeds "$SEEDS" \
    --gpus  "$GPUS" \
    --tag   "tremor_severe" \
    --variants "s2_w2_fs50_tremor_mod_severe"

echo ""
echo "========================================================"
echo "  All 5 configurations done!"
echo "  Finished : $(date)"
echo "========================================================"
