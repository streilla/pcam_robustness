import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from python_utils.datasets import PCAMDatasetFromPrecomputed
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train_labels_path', type=str, required=True)
parser.add_argument('--val_labels_path', type=str, required=True)
parser.add_argument('--train_features_path', type=str, default='precomputed_features/pcam_virchow2_train.h5')
parser.add_argument('--val_features_path', type=str, default='precomputed_features/pcam_virchow2_val.h5')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=1234)

args = parser.parse_args()

batch_size = args.batch_size
lr = args.lr
num_epochs = args.num_epochs
SEED = args.seed
train_labels = args.train_labels_path
val_labels = args.val_labels_path
train_features = args.train_features_path
val_features = args.val_features_path

train_params = {
    'train_labels': train_labels,
    'train_features': train_features,
    'val_labels': val_labels,
    'val_features':val_features,
    'batch_size': 64,
    'lr': 1e-5,
    'num_epochs': 100,
    'SEED': 1234,
}

Path('trained_models').mkdir(parents=True, exist_ok=True)
with open('trained_models/train_params.json', 'w') as f:
    json.dump(train_params, f)

# Set deterministic behavior
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# need to specify MLP layer and activation function for proper init
print('Loading Virchow2 patch encoder')
patch_encoder = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
patch_encoder = patch_encoder.eval()

transforms = create_transform(**resolve_data_config(patch_encoder.pretrained_cfg, model=patch_encoder))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
patch_encoder = patch_encoder.to(device)

train_loader = DataLoader(PCAMDatasetFromPrecomputed(train_features, train_labels), batch_size=batch_size)
val_loader = DataLoader(PCAMDatasetFromPrecomputed(val_features, val_labels), batch_size=1)

model = nn.Linear(in_features=2560, out_features=2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

min_val_loss = np.inf

for epoch in tqdm(range(num_epochs)):
    model.train()
    total_loss = 0.
    for features, labels, _ in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    model.eval()
    with torch.no_grad():
        total_val_loss = 0
        for features, labels, _ in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            val_loss = criterion(outputs, labels)
            total_val_loss += val_loss.item()
        if total_val_loss < min_val_loss:
            min_val_loss = total_val_loss
            torch.save(model.state_dict(), f"trained_models/best_virchow2.pt")
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Val loss: {total_val_loss/len(val_loader)}")