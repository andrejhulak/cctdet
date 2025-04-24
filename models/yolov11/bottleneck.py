import torch.nn as nn
from models.yolov11.convblock import Conv

class Bottleneck(nn.Module):
  def __init__(self, ic, oc, e, shortcut, *args, **kwargs):
    super().__init__(*args, **kwargs)
    hc = int(oc * e)
    self.shortcut = shortcut
    self.c1 = Conv(ic=ic, oc=hc, k=3, s=1, p=1, act=True)
    self.c2 = Conv(ic=hc, oc=oc, k=3, s=1, p=1, act=True)

  def forward(self, x):
    y = self.c2(self.c1(x))
    if self.shortcut:
      return x + y
    else:
      return y