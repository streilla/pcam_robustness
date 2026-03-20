import torch
from torchvision.transforms import v2, InterpolationMode
from python_utils.perturbations import PoissonNoise, BrightnessShift, GaussianNoise, JpegCompression


def get_transforms_fm(noise_type, noise_level):
    if noise_type == 'GaussianNoise':
        transforms = v2.Compose([
            GaussianNoise(std=noise_level),
            v2.Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
            v2.CenterCrop(size=(224,224)),
            v2.ToTensor(),
            v2.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
        ])
    elif noise_type == 'PoissonNoise':
        transforms = v2.Compose([
            PoissonNoise(n_photons=noise_level),
            v2.Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
            v2.CenterCrop(size=(224,224)),
            v2.ToTensor(),
            v2.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
        ])
    elif noise_type == 'BrightnessShift':
        transforms = v2.Compose([
            BrightnessShift(shift=noise_level),
            v2.Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
            v2.CenterCrop(size=(224,224)),
            v2.ToTensor(),
            v2.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
        ])
    elif noise_type == 'AugmentHE':
        transforms = v2.Compose([
            v2.Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
            v2.CenterCrop(size=(224,224)),
            v2.ToTensor(),
            v2.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
        ])
    elif noise_type == 'Jpeg':
        transforms = v2.Compose([
            JpegCompression(quality=noise_level),
            v2.Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
            v2.CenterCrop(size=(224,224)),
            v2.ToTensor(),
            v2.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
        ])
    else:
        raise NotImplementedError
    return transforms


def get_transforms_resnet(noise_type, noise_level, mean, std):
    if noise_type == 'GaussianNoise':
        transforms = v2.Compose([
                GaussianNoise(std=noise_level),
                v2.ToTensor(),
                v2.Normalize(mean=mean, std=std)
            ])
    elif noise_type == 'PoissonNoise':
        transforms = v2.Compose([
                PoissonNoise(n_photons=noise_level),
                v2.ToTensor(),
                v2.Normalize(mean=mean, std=std)
            ])
    elif noise_type == 'BrightnessShift':
        transforms = v2.Compose([
            BrightnessShift(shift=noise_level),
            v2.ToTensor(),
            v2.Normalize(mean=mean, std=std)
        ])
    elif noise_type == 'AugmentHE':
        transforms = v2.Compose([
            v2.Normalize(mean=mean, std=std) 
        ])
    elif noise_type == 'Jpeg':
        transforms = v2.Compose([
            JpegCompression(quality=noise_level),
            v2.ToTensor(),
            v2.Normalize(mean=mean, std=std)
        ])
    else:
        raise NotImplementedError('Unsupported noise type')
    return transforms