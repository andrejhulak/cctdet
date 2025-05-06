import torch
import os
from torchvision.io import read_file, decode_image
import torchvision

class VisDrone(torch.utils.data.Dataset):
  def __init__(self, root, transforms=None):
    super().__init__()
    self.root = root
    self.transforms = transforms

    self.images_dir = os.path.join(root, "images")
    self.annotations_dir = os.path.join(root, "annotations")
    self.imgs = sorted([
      f for f in os.listdir(self.images_dir)
      if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

  def __getitem__(self, idx):
    img_file = self.imgs[idx]
    img_path = os.path.join(self.images_dir, img_file)
    base_name = os.path.splitext(img_file)[0]

    img = decode_image(read_file(img_path)).to(torch.float32)
    # resize_transform = torchvision.transforms.Resize((640, 640))
    # img = resize_transform(img)

    ann_path = os.path.join(self.annotations_dir, f"{base_name}.txt")
    boxes = []
    labels = []

    if os.path.exists(ann_path):
      with open(ann_path, 'r') as f:
        for line in f:
          parts = line.strip().split(',')
          xmin = float(parts[0])
          ymin = float(parts[1])
          width = float(parts[2])
          height = float(parts[3])
          class_id = int(parts[5])
          
          if class_id == 0 or class_id == 11:
            continue

          class_id -= 1

          xmax = xmin + width
          ymax = ymin + height
          box = [xmin, ymin, xmax, ymax]

          boxes.append(box)
          labels.append(class_id)

    boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)
    target = { "boxes": boxes, "labels": labels }

    if self.transforms:
      img, target = self.transforms(img, target)

    # img /= 255.0

    return img, target, img_path

  def __len__(self):
    return len(self.imgs)