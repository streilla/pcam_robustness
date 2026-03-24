import numpy as np
import h5py
from tqdm import tqdm
from pathlib import Path
from python_utils.perturbations import JpegCompression, PoissonNoise, BrightnessShift, GaussianNoise
import os
import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--val_img_path', type=str, required=True)
parser.add_argument('--test_img_path', type=str, required=True)
#AugmentHE is treated differently
parser.add_argument('--noise_type', type=str, required=True, choices=['PoissonNoise', 'GaussianNoise', 'BrightnessShift', 'Jpeg'])

args = parser.parse_args()

with open('noise_levels.json') as f:
    noise_level_dict = json.load(f)
    
SEED = 1234
np.random.seed(SEED)

noise_dict = {"PoissonNoise": PoissonNoise,
            "GaussianNoise": GaussianNoise,
            "BrightnessShift": BrightnessShift,
            "Jpeg": JpegCompression}

noise_abvs = {"PoissonNoise": "poisson",
            "GaussianNoise": "gaussian",
            "BrightnessShift": "brightness",
            "Jpeg": "jpeg"}

val_imgs = args.val_img_path
test_imgs = args.test_img_path
noise_type = args.noise_type
noise = noise_dict[noise_type]
noise_levels = noise_level_dict[noise_type]
noise_abv = noise_abvs[noise_type]

multiply_naming = False
if noise_levels[1] < 1:
    multiply_naming = True

Path(noise_abv).mkdir(parents=True, exist_ok=True)

im_paths = [val_imgs, test_imgs]

for im_path, split in zip([val_imgs, test_imgs], ['val', 'test']):
    with h5py.File(im_path) as f:
        imgs = np.asarray(f['x'])
    
    for noise_level in noise_levels:
        if multiply_naming:
            #To handle floats in file name
            adj_noise_level = int(100*noise_level)
        else:
            adj_noise_level = noise_level
        if os.path.exists(f'{noise_abv}/{noise_abv}_{adj_noise_level}_{split}.h5'):
            print(f'Skipping level: {noise_level}')
            continue
        if noise_level == 0:
            with h5py.File(f'{noise_abv}/{noise_abv}_{adj_noise_level}_{split}.h5', 'w') as hf:
                hf.create_dataset("x",  data=imgs)
        else:
            augmented = []
            for img in tqdm(imgs):
                im_aug = noise(noise_level)(img)
                augmented.append(im_aug)
            with h5py.File(f'{noise_abv}/{noise_abv}_{adj_noise_level}_{split}.h5', 'w') as hf:
                hf.create_dataset("x",  data=augmented)