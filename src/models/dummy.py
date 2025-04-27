import torch
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DummyRandom(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    bs = len(x)
    outputs = []
    for img in x:
      c, h, w = img.shape
      n_obj = random.randint(0, 100)
      boxes = torch.rand((n_obj, 4), device=device)
      boxes[:, 0] *= w
      boxes[:, 1] *= h
      boxes[:, 2] *= w
      boxes[:, 3] *= h
      scores = torch.rand((n_obj), device=device)
      labels = torch.randint(0, 10, (n_obj,), device=device)
      outputs.append({'boxes' : boxes, 'scores' : scores, 'labels' : labels})
    return outputs