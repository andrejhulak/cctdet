import torch
import torch.nn as nn
from models.yolov11.convblock import Conv

class Detect(nn.Module):
  def __init__(self, ic, oc, nc, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.reg_conv = nn.Sequential(
      Conv(ic, oc, 3, 1, 1),
      Conv(oc, oc, 3, 1, 1),
      nn.Conv2d(oc, 5, 1, 1, 0)
    )
    
    self.cls_conv = nn.Sequential(
      Conv(ic, oc, 3, 1, 1),
      Conv(oc, oc, 3, 1, 1),
      nn.Conv2d(oc, nc, 1, 1, 0)
    )

  def forward(self, x):
    reg = self.reg_conv(x)
    cls = self.cls_conv(x)
    
    return torch.cat([
      reg.unsqueeze(1),
      cls.unsqueeze(1)
    ], dim=2).permute(0, 1, 3, 4, 2)