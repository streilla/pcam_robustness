import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import timm
from timm.layers import SwiGLUPacked
import numpy as np
import torch.nn as nn
from python_utils.datasets import PCAMDatasetFM, PCAMDataset
from python_utils.transforms import get_transforms_fm, get_transforms_resnet
import json
import argparse
from python_utils.inference_utils import eval_multiple 

parser = argparse.ArgumentParser()

parser.add_argument('--test_labels_path', type=str, required=True)
parser.add_argument('--val_labels_path', type=str, required=True)
parser.add_argument('--test_img_path', type=str, required=True)
parser.add_argument('--val_img_path', type=str, required=True)
parser.add_argument('--path_linear', type=str, default='trained_models/best_virchow2.pt')

args = parser.parse_args()

# need to specify MLP layer and activation function for proper init
print('Loading Virchow2 model')
patch_encoder = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
patch_encoder = patch_encoder.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
patch_encoder = patch_encoder.to(device)

val_ims = args.val_img_path
val_labels = args.val_labels_path

test_ims = args.test_img_path
test_labels = args.test_labels_path

print('Loading model')

def get_loader(model_type, im_path, label_path, noise_type, noise_level, device, patch_encoder=None):
    if model_type == 'virchow2':
        transforms = get_transforms_fm(noise_type, noise_level)
        loader = DataLoader(PCAMDatasetFM(im_path, label_path, patch_encoder, transforms, noise_type, device))
    elif model_type in ['res', 'res_lip']:
        #Computed on train set, with batch size of 128
        mean = [0.7007547616958618, 0.538357675075531, 0.6916205883026123]
        std = [0.23400866985321045, 0.27489325404167175, 0.21176908910274506]
        transforms = get_transforms_resnet(noise_type, noise_level, mean, std)
        loader = DataLoader(PCAMDataset(im_path, label_path, transforms, noise_type))
    else:
        raise NotImplementedError('Unsupported model type')
    return loader

def load_model(patch_encoder, device):
    if patch_encoder == 'virchow2':
        model = nn.Linear(2560, 2).to(device)
        best_checkpoint = 'trained_models/best_virchow2.pt'
    else:
        raise NotImplementedError('Unsupported patch encoder')
    model = model.to(device)
    state = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(state)
    return model

model_types = ['virchow2']
splits = ['val', 'test']

noise_types = ["PoissonNoise", "Jpeg", "GaussianNoise", "BrightnessShift"]
noise_dict = {"AugmentHE": 'augHE', "PoissonNoise": "poisson",
              "GaussianNoise": "gaussian", "BrightnessShift": "brightness",
              "Jpeg": "jpeg"}

#Loading noise levels for each noise type
with open('noise_levels.json', 'r') as f:
    noise_level_dict = json.load(f)

if __name__ == '__main__':
    for noise_type in noise_types:
        print(f'Noise - {noise_type}')
        noise_levels = noise_level_dict[noise_type]
        noise_abv = noise_dict[noise_type]
        for model_type in model_types:
            print(f'Model: {model_type}')
            model = load_model(model_type, device)
            for split in splits:
                if split == 'test':
                    ims = test_ims
                    labels = test_labels
                elif split == 'val':
                    ims = val_ims
                    labels = val_labels
                else:
                    raise NameError('Incorrect split')
                if os.path.exists(f'results/{noise_abv}/results_{noise_abv}_{split}_{model_type}.npy'):
                    print(f'Skipping {model_type} - {noise_type} - {split}')
                    continue
                all_scores = eval_multiple(model, model_type, ims, labels, noise_type, noise_levels, device, patch_encoder=patch_encoder, split=split)
                Path(f'results/{noise_abv}').mkdir(parents=True, exist_ok=True)
                np.save(f'results/{noise_abv}/results_{noise_abv}_{split}_{model_type}.npy', all_scores)