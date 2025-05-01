import torch
import torchvision.transforms.v2 as v2
import random

def get_transforms(img_size=(1333, 800), train=True):
  transforms_list = []

  if train:
    random_transforms = [
      v2.RandomAffine(
        degrees=(-30, 30),
        translate=(0.1, 0.1),
        scale=(0.8, 1.2),
        shear=(-5, 5),
        interpolation=v2.InterpolationMode.BILINEAR
      ),
      v2.RandomHorizontalFlip(p=0.5),
      v2.ColorJitter(hue=0.015, saturation=0.7, brightness=0.4),
      v2.RandomResizedCrop(size=img_size,
                            scale=(0.5, 1.0),
                            ratio=(0.75, 1.333),
                            antialias=True),
      v2.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)),
      v2.GaussianNoise()
    ]
    transforms_list.append(v2.RandomOrder([v2.RandomApply([t], p=0.8) for t in random_transforms]))
  else:
    transforms_list.append(v2.Resize(size=img_size, antialias=True))

  transforms_list.extend([
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  return v2.Compose(transforms_list)