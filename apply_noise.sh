#!/bin/bash

# ============================================================
# Save noised images
# Usage:
#   ./apply_noise.sh <VAL_IMG_PATH> <TEST_IMG_PATH> <INCLUDE_AUGHE>
#
# Example:
#   ./apply_noise.sh \
#     raw_data/camelyonpatch_level_2_split_valid_x.h5 \
#     raw_data/camelyonpatch_level_2_split_test_x.h5 \
#     true
# ============================================================

# Read parameters
VAL_IMG_PATH=$1
TEST_IMG_PATH=$2
INCLUDE_AUGHE=$3

# Check that required parameters are provided
if [ -z "$VAL_IMG_PATH" ] || [ -z "$TEST_IMG_PATH" ]; then
    echo "❌ Error: paths to VAL and TEST images are required."
    exit 1
fi

NOISE_TYPES=("GaussianNoise" "PoissonNoise" "BrightnessShift" "Jpeg")

for noise_type in "${NOISE_TYPES[@]}"; do
    echo "Applying ${noise_type}"
    python perturb_images.py --val_img_path $VAL_IMG_PATH --test_img_path $TEST_IMG_PATH --noise_type "GaussianNoise"
done

if $INCLUDE_AUGHE; then
    #This can take a while to run
    conda activate augHE
    echo "Applying AugmentHE"
    python apply_augHE.py --val_img_path $VAL_IMG_PATH --test_img_path $TEST_IMG_PATH
fi
# #Gaussian noise
# python perturb_images.py \
# --val_img_path $VAL_IMG_PATH \
# --test_img_path $TEST_IMG_PATH \
# --noise_type "GaussianNoise"

# #Poisson noise
# python perturb_images.py \
# --val_img_path $VAL_IMG_PATH \
# --test_img_path $TEST_IMG_PATH \
# --noise_type "PoissonNoise"

# #Brightness shift
# python perturb_images.py \
# --val_img_path $VAL_IMG_PATH \
# --test_img_path $TEST_IMG_PATH \
# --noise_type "BrightnessShift"

# #Jpeg compression
# python perturb_images.py \
# --val_img_path $VAL_IMG_PATH \
# --test_img_path $TEST_IMG_PATH \
# --noise_type "Jpeg"