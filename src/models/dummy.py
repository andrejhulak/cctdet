import torch

class Dummy(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    batch_s = len(x)