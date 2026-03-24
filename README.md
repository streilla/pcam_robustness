# PCAM robustness
Comparing classifier robustness to different types of noise on the PCAM dataset

### Installation
Download the PCAM data from https://github.com/basveeling/pcam and request Virchow2 access from https://huggingface.co/paige-ai/Virchow2

Create and activate a conda environment with
```
conda create -n pcam_robustness python=3.10
conda activate pcam_robustness
```

Install dependencies:
```
pip install -r requirements.txt
```

### Apply noise to images

For augmentHE noise, create a separate environment:
```
conda create -n augHE python=3.7
conda activate augHE
pip install -r requirements_augHE.txt
```

Troubleshooting: you may encounter difficulties with the installation of the spams library, if that happens, you can try:

```
conda install conda-forge::python-spams
```

To apply noise to the validation and test images, run
```
bash apply_noise.sh <VAL_IMG_PATH> <TEST_IMG_PATH> <INCLUDE_AUGHE>
```


### Run full pipeline
To train and test the pipeline with the linear classifier trained on features extracted with Virchow2:
```bash virchow_pipeline.sh <TRAIN_IMG_PATH> <TRAIN_LABELS_PATH> <VAL_IMG_PATH> <VAL_LABELS_PATH> <TEST_IMG_PATH> <TEST_LABELS_PATH>```

### Model training
Extract Virchow2 features with:
```
python encode_patches.py --train_img_path path/to/train_img --val_img_path path/to/val_img --test_img_path path/to/test_img --train_labels_path path/to/train_labels --val_labels_path path/to/val_labels --test_labels_path path/to/test_labels
```

Then, train a linear classifier with:
```
python train_model.py --train_labels_path path/to/train_labels --val_labels_path path/to/val_labels
```

For training the ResNet, 1-Lipschitz ResNet and MAGE models see https://github.com/camilodlt/evo_star_robustness_image_classification_submission

### DL Inference
To run inferences with the model trained with Virchow2 features on noised images, run
```python inference.py --test_labels_path path/to/test_labels --val_labels_path path/to/val_labels --test_img_path path/to/test_img --val_img_path path/to/val_img```

For inferences with ResNet and 1-Lipschitz ResNet, run
```python inference_resnet.py --test_labels_path path/to/test_labels --val_labels_path path/to/val_labels --test_img_path path/to/test_img --val_img_path path/to/val_img```

### MAGE setup
Install juliaup (https://github.com/JuliaLang/juliaup) then download version 1.11.7:
```
juliaup add 1.11.7~x64
juliaup default 1.11.7~x64
```

Activate MAGE environment with 
```
julia
using Pkg
Pkg.activate(".")
```

Download dependencies:
```
Pkg.add(url="https://github.com/camilodlt/SearchNetworks.jl.git")
Pkg.add(url="https://github.com/camilodlt/MAGE.jl.git")
Pkg.add(url="https://github.com/camilodlt/MAGE_SKIMAGE_MEASURE.git")
```

## MAGE inference

For MAGE inference, run the following command with a path to a python environment containing matplotlib and scikit-image:

```
export JULIA_PYTHONCALL_EXE="/path/to/python/env/w/matplotlib/and/skimage/python"
```

Then, run:

```
bash MAGE_inference.sh <VAL_LABELS_PATH> <TEST_LABELS_PATH> <OUTPUT_DIR> <TRIAL_ID>
```

