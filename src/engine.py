import torch
from torchmetrics.detection import MeanAveragePrecision

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.no_grad()
def evaluate(model, metric, dataloader):
  metric = MeanAveragePrecision().to(device)
  model.eval()
  metric.eval()

  for images, targets in dataloader:
    print(type(images))