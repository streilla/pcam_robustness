import os
import json
from pathlib import Path
from torchvision import models
import torch
import torch.nn as nn
import numpy as np
from python_utils.inference_utils import eval_multiple
import python_utils.lip_resnet as lip_resnet
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--test_labels_path', type=str, required=True)
parser.add_argument('--val_labels_path', type=str, required=True)
parser.add_argument('--test_img_path', type=str, required=True)
parser.add_argument('--val_img_path', type=str, required=True)
parser.add_argument('--path_resnet', type=str, default='trained_models/best_resnet.pth')
parser.add_argument('--path_reslip', type=str, default='trained_models/best_reslip.pth')

args = parser.parse_args()

def load_model(model_type, device):
    if model_type == 'res':
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )  # change channels if not RGB
        # LOAD WEIGHTS EVO*
        best_checkpoint = args.path_resnet
    elif model_type == 'res_lip':
        model = lip_resnet.lip_resnet18(num_classes = 2)
        # LOAD WEIGHTS EVO*     
        best_checkpoint = args.path_reslip
    model = model.to(device)
    state = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    return model

if __name__ == '__main__':
    val_ims = args.val_img_path
    val_labels = args.val_labels_path

    test_ims = args.test_img_path
    test_labels = args.test_labels_path
    
    model_types = ['res_lip', 'res']
    splits = ['val', 'test']
    model_type = 'res_lip'
    noise_types = ["PoissonNoise", "Jpeg", "GaussianNoise", "BrightnessShift"]
    noise_dict = {"AugmentHE": 'augHE', "PoissonNoise": "poisson",
                "GaussianNoise": "gaussian", "BrightnessShift": "brightness",
                "Jpeg": "jpeg"}

    with open('noise_levels.json', 'r') as f:
        noise_level_dict = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for noise_type in noise_types:
        print(f'Noise - {noise_type}')
        noise_levels = noise_level_dict[noise_type]
        noise_abv = noise_dict[noise_type]
        for model_type in model_types:
            print(f'Model: {model_type}')
            for split in splits:
                print(f'Split: {split}')
                if split == 'test':
                    ims = test_ims
                    labels = test_labels
                elif split == 'val':
                    ims = val_ims
                    labels = val_labels
                else:
                    raise NameError('Incorrect split')
                if os.path.exists(f'results/results_{noise_dict[noise_type]}_{split}_{model_type}.npy'):
                    print(f'Skipping: {model_type} - {noise_type} - {split}')
                    continue
                
                model = load_model(model_type, device)

                mean = [0.7007547616958618, 0.538357675075531, 0.6916205883026123]
                std = [0.23400866985321045, 0.27489325404167175, 0.21176908910274506]

                all_scores = eval_multiple(model, model_type, ims, labels,
                                             noise_type, noise_levels, device)
                Path(f'results/{noise_abv}').mkdir(parents=True, exist_ok=True)
                np.save(f'results/{noise_abv}/results_{noise_abv}_{split}_{model_type}.npy', all_scores)