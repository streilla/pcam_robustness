import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from python_utils.datasets import PCAMDatasetFM, PCAMDataset
from python_utils.transforms import get_transforms_fm, get_transforms_resnet

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

def eval_model(model, model_type, im_path, label_path, noise_type, noise_level, device, patch_encoder=None, split=None):
    if noise_type == 'AugmentHE':
        loader = get_loader(model_type=model_type, im_path=f'augHE/augHE_{noise_level}_{split}.h5',
                            label_path=label_path, noise_type=noise_type, noise_level=noise_level,
                            device=device, patch_encoder=patch_encoder)
    else:
        loader = get_loader(model_type=model_type, im_path=im_path,
                            label_path=label_path, noise_type=noise_type, noise_level=noise_level,
                            device=device, patch_encoder=patch_encoder)
    all_predicted = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for features, labels, _ in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            # Convert logits to probabilities and binary predictions
            predicted = torch.argmax(outputs, dim=1)
            all_predicted.append(predicted.cpu().detach())
            all_labels.append(labels.cpu().detach())
            
    score = balanced_accuracy_score(y_pred=all_predicted, y_true=all_labels)
    return score

def eval_multiple(model, model_type, im_path, label_path, noise_type, noise_levels, device, patch_encoder=None, split=None):
    # Set deterministic behavior
    SEED = 1234
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    all_scores = []
    for noise_level in tqdm(noise_levels):
        score = eval_model(model=model, model_type=model_type,
                           im_path=im_path, label_path=label_path,
                           noise_type=noise_type, noise_level=noise_level, device=device,
                           patch_encoder=patch_encoder, split=split)
        all_scores.append(score)
    return np.array(all_scores)