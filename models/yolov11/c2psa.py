import torch
import torch.nn as nn
from models.yolov11.convblock import Conv
from models.yolov11.psablock import PSABlock

class C2PSA(torch.nn.Module):
  def __init__(self, ic, oc, n):
    super().__init__()
    self.oc = oc
    self.conv1 = Conv(ic, oc, p=0)
    self.psa = torch.nn.Sequential(*(PSABlock(oc // 2, oc // 128) for _ in range(n)))
    self.conv2 = Conv(oc, oc, p=0)

  def forward(self, x):
    chunk0, chunk1 = self.conv1(x).chunk(2, dim=1)
    return self.conv2(torch.cat(tensors=(chunk0, self.psa(chunk1)), dim=1))