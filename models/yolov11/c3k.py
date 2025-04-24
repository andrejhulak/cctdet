import torch.nn as nn
import torch
from models.yolov11.bottleneck import Bottleneck
from models.yolov11.convblock import Conv

class C3K(nn.Module):
  def __init__(self, ic, oc, n, shortcut, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.c1_1 = Conv(ic=ic, oc=(oc//2), k=1, s=1, p=0, act=True)
    self.c1_2 = Conv(ic=ic, oc=(oc//2), k=1, s=1, p=0, act=True)

    self.bottlenecks = nn.ModuleList(
      Bottleneck(ic=(oc//2), oc=(oc//2), e=1.0, shortcut=shortcut)
      for _ in range(n)
    )

    self.c2 = Conv(ic=oc, oc=oc, k=1, s=1, p=0, act=True)

  def forward(self, x):
    y = self.c1_1(x)

    x = self.c1_2(x)

    for b in self.bottlenecks:
      x = b(x)

    x = torch.cat((y, x), dim=1)

    x = self.c2(x)
    return x