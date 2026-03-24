#!/usr/bin/env bash

# ============================================================
# Run MAGE experiments for all noise types
# Usage:
#   ./MAGE_inference.sh <VAL_LABELS_PATH> <TEST_LABELS_PATH> <OUTPUT_DIR> <TRIAL_ID>
#
# Example:
#   ./MAGE_inference.sh \
#     raw_data/camelyonpatch_level_2_split_valid_y.h5 \
#     raw_data/camelyonpatch_level_2_split_test_y.h5 \
#     mage_results \
#     EXP_1 \
# ============================================================

export UTCGP_CONSTRAINED=yes
export UTCGP_MIN_INT=-10000
export UTCGP_MAX_INT=10000
export UTCGP_MIN_FLOAT=-10000
export UTCGP_MAX_FLOAT=10000
export UTCGP_SMALL_ARRAY=100
export UTCGP_BIG_ARRAY=1000

# Read parameters or use defaults
VAL_LABELS_PATH=$1
TEST_LABELS_PATH=$2
OUTPUT_DIR=$3
TRIAL_ID=$4
ACT="identity"
USE_SKI="true"

# Number of Julia threads
THREADS=16

# Check required parameters
if [ -z "$VAL_LABELS_PATH" ] || [ -z "$TEST_LABELS_PATH" ]; then
    echo "❌ Paths to val and test data and labels are required."
    exit 1
fi

if [ -z "$OUTPUT_DIR" ] || [ -z "$TRIAL_ID" ]; then
    echo "❌ OUTPUT_DIR, and TRIAL_ID are required."
    exit 1
fi

# Model directory
MODEL_DIR="results_pcam_rgb_mage/EXP_1/mage_imgcls"

# Dataset directory
NOISE_TYPES=("augHE" "gaussian" "poisson" "brightness" "jpeg")

# Print configuration
echo "============================================================"
echo " Running MAGE inference with parameters:"
echo " NOISE_TYPES  = ${NOISE_TYPES[@]}"
echo " OUTPUT_DIR  = $OUTPUT_DIR"
echo " TRIAL_ID    = $TRIAL_ID"
echo " ACT         = $ACT"
echo " USE_SKI     = $USE_SKI"
echo "============================================================"

for noise_type in "${NOISE_TYPES[@]}"; do
    echo "------------------------------------------------------------"
    echo " Running experiments for ${noise_type}"
    echo "------------------------------------------------------------"

    if ! [ -d "${OUTPUT_DIR}/${noise_type}"]; then
        mkdir "${OUTPUT_DIR}/${noise_type}"
    fi

    CMD=(./mage_runs/run_${noise_type}_tests.sh
        $VAL_LABELS_PATH
        $TEST_LABELS_PATH
        "${OUTPUT_DIR}/${noise_type}"
        $TRIAL_ID
        # "val test"
        "test"
    )

    # Run the command
    "${CMD[@]}"
done