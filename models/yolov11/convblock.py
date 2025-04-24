import torch.nn as nn

class Conv(nn.Module):
  def __init__(self, ic, oc, k=1, s=1, p=1, act=True, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.ic = ic
    self.oc = oc

    self.c = nn.Conv2d(in_channels=ic,
                        out_channels=oc,
                        kernel_size=k,
                        stride=s,
                        padding=p,
                        bias=False)
    self.bn = nn.BatchNorm2d(num_features=oc, eps=0.001, momentum=0.03)
    self.a = nn.SiLU() if act else nn.Identity()

  def forward(self, x):
    return self.a(self.bn(self.c(x)))