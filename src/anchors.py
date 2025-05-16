from datasets.ds import VisDrone
from torch.utils.data import DataLoader
from utils.misc import collate_fn_simple
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torchvision.transforms.v2 as v2

BATCH_SIZE = 1
k = 9

resize_transform = v2.Resize(size=(800, 1333), antialias=True)

if __name__ == "__main__":
  val_root = "data/VisDrone/VisDrone2019-DET-val"
  val_ds = VisDrone(root=val_root, transforms=resize_transform)
  val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                      collate_fn=collate_fn_simple)

  all_boxes = []

  for imgs, targets, _ in tqdm(val_dl):
    h, w = imgs[0].shape[1:]
    boxes = targets[0]['boxes']
    widths = (boxes[:, 2] - boxes[:, 0]) / w
    heights = (boxes[:, 3] - boxes[:, 1]) / h
    wh = torch.stack((widths, heights), dim=1)
    all_boxes.append(wh)

  all_boxes = torch.cat(all_boxes, dim=0).numpy()
  print(f"Total boxes: {all_boxes.shape[0]}")

  kmeans = KMeans(n_clusters=k, random_state=0).fit(all_boxes)
  anchors = kmeans.cluster_centers_

  anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]

  print("\nAnchor boxes (w, h) normalized:")
  for i, (w, h) in enumerate(anchors):
    print(f"Anchor {i+1}: ({w:.4f}, {h:.4f})")

  plt.figure(figsize=(6, 6))
  plt.scatter(all_boxes[:, 0], all_boxes[:, 1], alpha=0.05, label='GT boxes')
  plt.scatter(anchors[:, 0], anchors[:, 1], c='red', marker='x', s=100, label='Anchors')
  plt.xlabel("Normalized Width")
  plt.ylabel("Normalized Height")
  plt.legend()
  plt.title("Anchor Box Clustering")
  plt.grid(True)
  plt.show()