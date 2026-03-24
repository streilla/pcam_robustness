#!/usr/bin/env bash

# ============================================================
# Run JPEG experiments for MAGE models
# Usage:
#   ./run_jpeg_tests.sh <VAL_LABELS> <TEST_LABELS> <OUTPUT_DIR> <TRIAL_ID> <MODES>
#
# Example:
#   ./run_jpeg_tests.sh \
#     raw_data/camelyonpatch_level_2_split_valid_y.h5 \
#     raw_data/camelyonpatch_level_2_split_test_y.h5 \
#     mage_results \
#     EXP_1 \
#     "val test"
# ============================================================

# Read parameters or use defaults
VAL_LABELS=$1
TEST_LABELS=$2
OUTPUT_DIR=$3
TRIAL_ID=$4
MODES=$5
ACT="identity"
USE_SKI="true"


# Number of Julia threads
THREADS=16

# Check required parameters
if [ -z "$OUTPUT_DIR" ] || [ -z "$TRIAL_ID" ]; then
    echo "❌ OUTPUT_DIR, and TRIAL_ID are required."
    exit 1
fi

if [ -z "$MODES" ]; then
    echo "❌ Error: you must specify at least one mode (val, test)."
    exit 1
fi

# Validate modes
for mode in $MODES; do
    if [[ ! "$mode" =~ ^(val|test)$ ]]; then
        echo "❌ Invalid mode: $mode (must be one of: val, test)"
        exit 1
    fi
done

# Model directory
MODEL_DIR="results_pcam_rgb_mage/EXP_1/mage_imgcls"

# Dataset directory
DATA_DIR="jpeg"

# Jpeg noise intensity levels
JPEG_LEVELS=("0" "90" "80" "70" "60" "50" "40" "30" "20" "10" "9" "8" "7" "6" "5" "4" "3" "2" "1") 

MODEL_EXPS=("EXP_1__13" "EXP_1__21" "EXP_1__20" "EXP_1__15" "EXP_1__14")

# Print configuration
echo "============================================================"
echo " Running JPEG tests with parameters:"
echo " MODEL_EXP  = ${MODEL_EXPS[@]}"
echo " OUTPUT_DIR  = $OUTPUT_DIR"
echo " TRIAL_ID    = $TRIAL_ID"
echo " ACT         = $ACT"
echo " USE_SKI     = $USE_SKI"
echo " MODES       = $MODES"
echo " JPEG  = ${JPEG_LEVELS[@]}"
echo "============================================================"

# Main loop
for exp in "${MODEL_EXPS[@]}"; do
    MODEL_PATH="${MODEL_DIR}/${exp}/checkpoint_0.pickle"
    for level in "${JPEG_LEVELS[@]}"; do
        if [ -z "$level" ]; then
            VAL_DATA="${DATA_DIR}/${DATA_DIR}_0_val.h5"
            TEST_DATA="${DATA_DIR}/${DATA_DIR}_0_test.h5"
            LABEL="no noise"
        else
            VAL_DATA="${DATA_DIR}/${DATA_DIR}_${level}_val.h5"
            TEST_DATA="${DATA_DIR}/${DATA_DIR}_${level}_test.h5"
            LABEL="${DATA_DIR}_${level}"
        fi

        echo "------------------------------------------------------------"
        echo " Running experiment ${exp} for ${LABEL} (${MODES})"
        echo "------------------------------------------------------------"

        # Build the Julia command dynamically based on selected modes
        CMD=(julia --project -t $THREADS test_noise_mage.jl
            --model_path="$MODEL_PATH"
            --output_dir="$OUTPUT_DIR"
            --trial_id="$TRIAL_ID"
            --act="$ACT"
            --use_ski="$USE_SKI"
            --noise_level="$level"
        )

        [[ "$MODES" =~ "val" ]]   && CMD+=(--val_data_location="$VAL_DATA") && CMD+=(--val_labels_location="$VAL_LABELS")
        [[ "$MODES" =~ "test" ]]  && CMD+=(--test_data_location="$TEST_DATA") && CMD+=(--test_labels_location="$TEST_LABELS")

        # Run the command
        "${CMD[@]}"
    done
done