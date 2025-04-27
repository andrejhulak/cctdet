import torch
from torchmetrics.detection import MeanAveragePrecision

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