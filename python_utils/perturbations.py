import torch
from torchvision.transforms import v2
import numpy as np
from PIL import Image
from io import BytesIO

class PoissonNoise(torch.nn.Module):
    def __init__(self, n_photons):
        super().__init__()
        self.n_photons = n_photons
        
    def forward(self, img):
        if self.n_photons == 0:
            return Image.fromarray(img)
        else:
            n = np.max(img)
            lam = img / n * self.n_photons
            new_img = np.uint8(np.clip(np.random.poisson(lam) / self.n_photons * n, 0, 255))
            return Image.fromarray(new_img)
        
class GaussianNoise(torch.nn.Module):
    def __init__(self, std):
        super().__init__()
        self.std = std
        
    def forward(self, img):
        noise_im = np.random.normal(loc=0, scale=self.std*255, size=(96, 96, 3))
        new_img = np.uint8(np.clip(img + noise_im, 0, 255))
        return Image.fromarray(new_img)
    
class BrightnessShift(torch.nn.Module):
    def __init__(self, shift):
        super().__init__()
        self.shift = shift
        
    def forward(self, img):
        new_img = np.uint8(np.clip(img + (self.shift * 255) * np.ones((96,96,3)), 0, 255))
        return Image.fromarray(new_img)
    
class JpegCompression(torch.nn.Module):
    def __init__(self, quality):
        super().__init__()
        self.quality = quality
        
    def forward(self, img):
        if self.quality == 0:
            return Image.fromarray(img)
        else:
            buffer = BytesIO()
            im = Image.fromarray(img)
            im.save(buffer, "JPEG", quality=self.quality)
            compressed_data = buffer.getvalue()
            buffer.close()
            new_img = Image.open(BytesIO(compressed_data))
            return new_img