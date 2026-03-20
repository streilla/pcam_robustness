import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
import h5py
from PIL import Image
import numpy as np
 
class PCAMDataset(Dataset):
    def __init__(self, im_path, label_path, transforms, noise_type=None):
        self.im_path = im_path
        self.label_path = label_path
        self.transforms = transforms
        self.noise_type = noise_type
            
    def __len__(self):
        with h5py.File(self.label_path, "r") as f:
            label = np.asarray(f["y"][:])
        return len(label)

    def __getitem__(self, idx):
        with h5py.File(self.im_path, "r") as f:
            im = np.asarray(f["x"][idx])
        with h5py.File(self.label_path, "r") as f:
            label = np.asarray(f["y"][idx]).squeeze()
            
        if self.noise_type is None:
            #conversion otherwise occurs when applying noise
            im = Image.fromarray(im)
            
        if self.noise_type == 'AugmentHE':
            #conversion otherwise occurs when applying noise
            im = pil_to_tensor(Image.fromarray(im))
            im = im.type(torch.float32)
            im = im / 255.0
        
        im = self.transforms(im)
        label = torch.tensor(label, dtype=torch.long)
        return im, label, idx


class PCAMDatasetFM(Dataset):
    def __init__(self, im_path, label_path, patch_encoder, transforms, noise_type, device):
        self.im_path = im_path
        self.label_path = label_path
        self.transforms = transforms
        self.patch_encoder = patch_encoder
        self.noise_type = noise_type
        self.device = device
            
    def __len__(self):
        with h5py.File(self.label_path, "r") as f:
            label = np.asarray(f["y"][:])
        return len(label)

    def __getitem__(self, idx):
        with h5py.File(self.im_path, "r") as f:
            im = np.asarray(f["x"][idx])
        with h5py.File(self.label_path, "r") as f:
            label = np.asarray(f["y"][idx]).squeeze()
        
        im = self.transforms(im).unsqueeze(0)
        im = im.to(self.device)
        output = self.patch_encoder(im)
        class_token = output[:, 0]    # size: 1 x 1280
        patch_tokens = output[:, 5:]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those

        # concatenate class token and average pool of patch tokens
        features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560
        features = features.squeeze()
        label = torch.tensor(label, dtype=torch.long)
        return features, label, idx
    
    
class PCAMDatasetFromPrecomputed(Dataset):
    def __init__(self, features_path, label_path):
        self.features_path = features_path
        self.label_path = label_path
            
    def __len__(self):
        with h5py.File(self.label_path, "r") as f:
            labels = np.asarray(f["y"])
        return len(labels)

    def __getitem__(self, idx):
        with h5py.File(self.features_path, "r") as f:
            feature = np.asarray(f["features"][idx])
        with h5py.File(self.label_path, "r") as f:
            label = np.asarray(f["y"][idx]).squeeze()
        label = torch.tensor(label, dtype=torch.long)
        return feature, label, idx
