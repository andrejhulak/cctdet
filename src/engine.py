import torch
from torchmetrics.detection import MeanAveragePrecision
from utils.misc import class_names
import torchvision.transforms.functional as F
import cv2
import numpy as np
from tqdm import tqdm
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, dataloader, optimizer, epoch, save_dir="model_weights/cctdetnew/weights_0"):
  model.train()
  total_loss = 0.0
  total_losses = {}
  progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  for batch_idx, (images, targets) in enumerate(progress_bar):
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    # with torch.autocast(device_type='cuda', dtype=torch.float16):
    losses = model(images, targets)
    loss = sum(l for l in losses.values())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    for k, v in losses.items():
      total_losses[k] = total_losses.get(k, 0.0) + v.item()

    avg_loss = total_loss / (batch_idx + 1)
    postfix = {"avg_loss": f"{avg_loss:.4f}"}

    postfix.update({k: f"{(total_losses[k]/(batch_idx+1)):.4f}" for k in total_losses})
    progress_bar.set_postfix(postfix)

  save_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")

  torch.save(model.state_dict(), save_path)
  print(f"Saved model parameters to: {save_path}")

  return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader):
  model.eval()
  metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True).to(device)
  
  for images, targets, _ in tqdm(dataloader):
    imgs = [img.to(device) for img in images]
    
    detections = model(imgs)
    
    preds = []
    for i, det in enumerate(detections):
      preds.append({
        'boxes': det['boxes'].detach().cpu(),
        'scores': det['scores'].detach().cpu(),
        'labels': det['labels'].detach().cpu().int()
      })

    gt_boxes = [t['boxes'].cpu() for t in targets]
    gt_labels = [t['labels'].flatten().cpu().int() for t in targets]
    print(f'gt_labels min: {min(gt_labels[i])}')
    print(f'gt_labels max: {max(gt_labels[i])}')

    print(f'pred_labels min: {min(preds[i]["labels"])}')
    print(f'pred_labels max: {max(preds[i]["labels"])}')

    gt = [{'boxes': b, 'labels': l} for b, l in zip(gt_boxes, gt_labels)]

    metric.update(preds, gt)
  
  return metric.compute()

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