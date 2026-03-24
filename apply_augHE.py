import os
import json
import argparse
import numpy as np
import h5py
from tqdm import tqdm
from pathlib import Path
from python_utils.augmentHE import StainAugmentor

parser = argparse.ArgumentParser()

parser.add_argument('--val_img_path', type=str, required=True)
parser.add_argument('--test_img_path', type=str, required=True)

args = parser.parse_args()

with open('noise_levels.json') as f:
    noise_level_dict = json.load(f)
    
SEED = 1234
np.random.seed(SEED)

val_imgs = args.val_img_path
test_imgs = args.test_img_path
noise_type = 'AugmentHE'
noise = StainAugmentor
noise_levels = noise_level_dict[noise_type]
noise_abv = 'augHE'

multiply_naming = False
if noise_levels[1] < 1:
    multiply_naming = True

Path(noise_abv).mkdir(parents=True, exist_ok=True)

im_paths = [val_imgs, test_imgs]

for im_path, split in zip([val_imgs, test_imgs], ['val', 'test']):
    with h5py.File(im_path) as f:
        imgs = np.asarray(f['x'])
    
    for noise_level in noise_levels:
        print(f'Noise level {noise_level}')
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
            aug = StainAugmentor(alpha_range=noise_level,
                                    beta_range=noise_level,
                                    alpha_stain_range=noise_level,
                                    beta_stain_range=noise_level,
                                    seed=SEED)
            for img in tqdm(imgs):
                try:
                    img_aug = aug(image=img)["image"]
                    img_aug = np.uint8(np.clip(255*img_aug, 0, 255))
                except:
                    img_aug = img
                augmented.append(img_aug)
            with h5py.File(f'{noise_abv}/{noise_abv}_{adj_noise_level}_{split}.h5', 'w') as hf:
                hf.create_dataset("x",  data=augmented)