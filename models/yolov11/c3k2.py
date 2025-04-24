import torch.nn as nn
from models.yolov11.convblock import Conv
from models.yolov11.c3k import C3K
from models.yolov11.bottleneck import Bottleneck
import torch

class C3K2(nn.Module):
  def __init__(self, ic, oc, c3k, n, e, shortcut=True):
    super().__init__()
    self.ic = ic
    self.oc = oc
    self.c3k = c3k
    self.n = n
    self.e = e
    self.shortcut = shortcut

    self.c1 = Conv(ic=ic, oc=(self.n+1) * int(self.oc * self.e), k=1, s=1, p=0)

    if self.c3k:
      self.c3k_block = C3K(
        ic=int(self.oc * self.e),
        oc=int(self.oc * self.e),
        n=self.n,
        shortcut=self.shortcut
      )
    else:
      self.bottlenecks = nn.ModuleList([
        Bottleneck(
          ic=int(self.oc * self.e),
          oc=int(self.oc * self.e),
          e=self.e,
          shortcut=self.shortcut
        )
        for _ in range(n)
      ])

    self.c2 = Conv(ic=(self.n + 2) * int(self.oc * self.e), oc=self.oc, k=1, s=1, p=0)

  def forward(self, x):
    x = self.c1(x)

    if self.c3k:
      chunks = x.chunk(self.n + 1, dim=1)
      chunk0 = chunks[0]
      chunk1 = chunks[1]

      c3k_out = self.c3k_block(chunk1)

      if len(chunks) > 2:
        out = torch.cat([chunk0, chunk1, c3k_out, chunks[2:][0]], dim=1)
      else:
        out = torch.cat([chunk0, chunk1, c3k_out], dim=1)
      return self.c2(out)
    else:
      chunks = x.chunk(self.n + 1, dim=1)
      chunk0 = chunks[0]
      chunk1 = chunks[1]
      out = [chunk0]
      out.append(self.bottlenecks[0](chunk1))

      t = out[-1]
      for b in self.bottlenecks:
          t = b(t)
      out.append(t)

      if len(chunks) > 2:
        out.append(chunks[2:][0])

      x_cat = torch.cat(out, dim=1)
      return self.c2(x_cat)