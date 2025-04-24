import torch.nn as nn
import torch
from models.yolov11.convblock import Conv

class SPPF(nn.Module):
  def __init__(self, ic, oc, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.oc = oc
    self.c1 = Conv(ic=ic, oc=oc, k=3, s=1)
    self.mp = nn.MaxPool2d(kernel_size=5, stride=1, padding=5//2)
    self.c2 = Conv(ic=4*oc, oc=oc, k=3, s=1)

  def forward(self, x):
    x = self.c1(x)
    y1 = self.mp(x)
    y2 = self.mp(y1)
    y3 = self.mp(y2)
    out = torch.cat([x, y1, y2, y3], dim=1)
    return self.c2(out)