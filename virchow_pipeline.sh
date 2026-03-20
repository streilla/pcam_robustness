#!/bin/bash

# ============================================================
# Run noise experiments for Virchow2 models
# Usage:
#   ./virchow_pipeline.sh <TRAIN_IMG_PATH> <TRAIN_LABELS_PATH> <VAL_IMG_PATH> <VAL_LABELS_PATH> <TEST_IMG_PATH> <TEST_LABELS_PATH>
#
# Example:
#   ./virchow_pipeline.sh \
#     raw_data/camelyonpatch_level_2_split_train_x.h5 \
#     raw_data/camelyonpatch_level_2_split_train_y.h5 \
#     raw_data/camelyonpatch_level_2_split_valid_x.h5 \
#     raw_data/camelyonpatch_level_2_split_valid_y.h5 \
#     raw_data/camelyonpatch_level_2_split_test_x.h5 \
#     raw_data/camelyonpatch_level_2_split_test_y.h5
# ============================================================

# Read parameters
TRAIN_IMG_PATH=$1
TRAIN_LABELS_PATH=$2
VAL_IMG_PATH=$3
VAL_LABELS_PATH=$4
TEST_IMG_PATH=$5
TEST_LABELS_PATH=$6

# Check that required parameters are provided
if [ -z "$TRAIN_IMG_PATH" ] || [ -z "$TRAIN_LABELS_PATH" ] || [ -z "$VAL_IMG_PATH" ] || [ -z "$VAL_LABELS_PATH" ] || [ -z "$TEST_IMG_PATH" ] || [ -z "$TEST_LABELS_PATH" ]; then
    echo "❌ Error: paths to TRAIN, VAL and TEST images and labels are required."
    exit 1
fi

#Extracting patch feature representations for model training
python encode_patches.py \
--train_img_path $TRAIN_IMG_PATH \
--train_labels_path $TRAIN_LABELS_PATH \
--val_img_path $VAL_IMG_PATH \
--val_labels_path $VAL_LABELS_PATH \
--test_img_path $TEST_IMG_PATH \
--test_labels_path $TEST_LABELS_PATH

#Model training
python train_model.py \
--train_labels_path $TRAIN_LABELS_PATH \
--val_labels_path $VAL_LABELS_PATH

#Inference
python inference.py \
--val_img_path $VAL_IMG_PATH \
--val_labels_path $VAL_LABELS_PATH \
--test_img_path $TEST_IMG_PATH \
--test_labels_path $TEST_LABELS_PATH