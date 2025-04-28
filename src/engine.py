import torch
from torchmetrics.detection import MeanAveragePrecision
from utils.misc import class_names
import torchvision.transforms.functional as F
import cv2
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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