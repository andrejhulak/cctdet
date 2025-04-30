import torch
from torchmetrics.detection import MeanAveragePrecision
from utils.misc import class_names
import torchvision.transforms.functional as F
import cv2
import numpy as np
from tqdm import tqdm
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, dataloader, optimizer, epoch, save_dir="model_weights/fasterrcnn/weights"):
  model.train()
  total_loss = 0.0
  num_batches = len(dataloader)
  progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  for images, targets in progress_bar:
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    losses = model(images, targets)
    loss = sum(l for l in losses.values())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    loss_dict_str = ", ".join(f"{k}: {v.item():.4f}" for k, v in losses.items())
    progress_bar.set_postfix({"loss": f"{loss.item():.4f}", **{k: f"{v.item():.4f}" for k, v in losses.items()}})

  save_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
  torch.save(model.state_dict(), save_path)
  print(f"Saved model parameters to: {save_path}")

  return total_loss / num_batches

@torch.no_grad()
def evaluate(model, dataloader):
  metric = MeanAveragePrecision(box_format='xywh', class_metrics=True).to(device)
  model.eval()
  metric.eval()
  ret = None

  for images, targets in dataloader:
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    outputs = model(images)

    metric.update(outputs, targets)

  ret = metric.compute()
  return ret

# don't think this works
@torch.no_grad()
def visualize_image(model, image, class_names=class_names, min_score_thresh=0.5):
  model.eval()
  prediction = model([image])[0]

  img = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

  boxes = prediction['boxes'].cpu().numpy().astype(np.int32)
  labels = prediction['labels'].cpu().numpy()
  scores = prediction['scores'].cpu().numpy()

  for box, label, score in zip(boxes, labels, scores):
    if score > min_score_thresh:
      xmin, ymin, xmax, ymax = box
      cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
      class_name = class_names.get(label.item(), f'Class {label.item()}')
      caption = f'{class_name}: {score:.2f}'
      cv2.putText(img, caption, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

  cv2.imshow('Prediction', img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()