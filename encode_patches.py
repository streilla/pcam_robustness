from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import h5py
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
import numpy as np
from python_utils.datasets import PCAMDataset
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train_img_path', type=str, required=True)
parser.add_argument('--train_labels_path', type=str, required=True)
parser.add_argument('--val_img_path', type=str, required=True)
parser.add_argument('--val_labels_path', type=str, required=True)
parser.add_argument('--test_img_path', type=str, required=True)
parser.add_argument('--test_labels_path', type=str, required=True)

args = parser.parse_args()

# need to specify MLP layer and activation function for proper init
print('Loading Virchow2 encoder')
model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
model = model.eval()

transforms = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

train_ims = args.train_img_path
train_labels = args.train_labels_path

val_ims = args.val_img_path
val_labels = args.val_labels_path

test_ims = args.test_img_path
test_labels = args.test_labels_path

train_loader = DataLoader(PCAMDataset(train_ims, train_labels, transforms))
val_loader = DataLoader(PCAMDataset(val_ims, val_labels, transforms))
test_loader = DataLoader(PCAMDataset(test_ims, test_labels, transforms))

loaders = [train_loader, val_loader, test_loader]

for loader, split in zip(loaders, ['train', 'val', 'test']):
    embeddings = []
    for image, label, idx in tqdm(loader):
        #image = transforms(image).unsqueeze(0)  # size: 1 x 3 x 224 x 224
        image = image.to(device)
        output = model(image)  # size: 1 x 261 x 1280

        class_token = output[:, 0]    # size: 1 x 1280
        patch_tokens = output[:, 5:]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those

        # concatenate class token and average pool of patch tokens
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560
        embeddings.append(embedding.cpu().detach().numpy())
    
    embeddings = np.array(embeddings).squeeze()
    Path('precomputed_features').mkdir(parents=True, exist_ok=True)
    with h5py.File(f'precomputed_features/pcam_virchow2_{split}.h5', 'w') as hf:
        hf.create_dataset("features",  data=embeddings)
    