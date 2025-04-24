import torch.nn as nn
import torch
from models.yolov11.convblock import Conv
from models.yolov11.c3k2 import C3K2
from models.yolov11.sppf import SPPF
from models.yolov11.c2psa import C2PSA
from models.yolov11.detect import Detect

class Yolo11(nn.Module):
  def __init__(self, d=0.5, w=0.5, mc=1024):
    super().__init__()
    self.c1 = Conv(ic=3, oc=int(min(64, mc)*w), k=3, s=2)
    self.c2 = Conv(ic=self.c1.oc, oc=int(min(128, mc)*w), k=3, s=2)
    self.c3k2_1 = C3K2(ic=self.c2.oc, oc=int(min(256, mc)*w), n=int(2*d), c3k=False, e=0.25)

    self.c3 = Conv(ic=self.c3k2_1.oc, oc=int(min(256, mc)*w), k=3, s=2)
    self.c3k2_2 = C3K2(ic=self.c3.oc, oc=int(min(512, mc)*w), n=int(2*d), c3k=False, e=0.25)

    self.c4 = Conv(ic=self.c3k2_2.oc, oc=int(min(512, mc)*w), k=3, s=2)
    self.c3k2_3 = C3K2(ic=self.c4.oc, oc=int(min(1024, mc)*w), n=int(2*d), c3k=True, e=1)

    self.c5 = Conv(ic=self.c3k2_3.oc, oc=int(min(1024, mc)*w), k=3, s=2)
    self.c3k2_4 = C3K2(ic=self.c5.oc, oc=int(min(1024, mc)*w), n=int(2*d), c3k=True, e=1)

    self.sppf = SPPF(ic=self.c3k2_4.oc, oc=int(min(1024, mc)*w))
    self.c2psa = C2PSA(ic=self.sppf.oc, oc=int(min(1024, mc)*w), n=int(2*d))

    self.u = nn.Upsample(scale_factor=2)

    self.c3k2_5 = C3K2(ic=self.c2psa.oc + self.c3k2_3.oc, oc=int(min(512, mc)*w), n=int(2*d), c3k=False, e=1)

    self.c3k2_6 = C3K2(ic=self.c3k2_5.oc + self.c3k2_2.oc, oc=int(min(256, mc)*w), n=int(2*d), c3k=False, e=1)

    self.c6 = Conv(ic=self.c3k2_6.oc, oc=int(min(256, mc)*w), k=3, s=2)

    self.c3k2_7 = C3K2(ic=self.c6.oc + self.c3k2_5.oc, oc=int(min(512, mc)*w), n=int(2*d), c3k=False, e=1)

    self.c7 = Conv(ic=self.c3k2_7.oc, oc=int(min(512, mc)*w), k=3, s=2)

    self.c3k2_8 = C3K2(ic=self.c2psa.oc + self.c7.oc, oc=int(min(1024, mc)*w), n=int(2*d), c3k=False, e=1)

    self.detect_top = Detect(ic=self.c3k2_6.oc, oc=mc//2, nc=11)
    self.detect_mid = Detect(ic=self.c3k2_7.oc, oc=mc//2, nc=11)
    self.detect_bot = Detect(ic=self.c3k2_8.oc, oc=mc//2, nc=11)

  def forward(self, x):
    # backbone
    x = self.c3k2_1(self.c2(self.c1(x)))
    x_top = self.c3k2_2(self.c3(x))

    x_mid = self.c3k2_3(self.c4(x_top))

    x_bot = self.c3k2_4(self.c5(x_mid))

    # neck
    x_bot = self.c2psa(self.sppf(x_bot))

    x_temp = self.u(x_bot)
    x_second = torch.cat([x_mid, x_temp], dim=1)
    x_second = self.c3k2_5(x_second)

    x_temp_2 = self.u(x_second)
    x_first = torch.cat([x_top, x_temp_2], dim=1)
    x_ready_top = self.c3k2_6(x_first)

    x_temp_3 = torch.cat([x_second, self.c6(x_ready_top)], dim=1)
    x_ready_mid = self.c3k2_7(x_temp_3)

    x_temp_4 = torch.cat([x_bot, self.c7(x_ready_mid)], dim=1)
    x_ready_bot = self.c3k2_8(x_temp_4)

    #head
    ret_top = self.detect_top(x_ready_top)
    ret_mid = self.detect_mid(x_ready_mid)
    ret_bot = self.detect_bot(x_ready_bot)

    return (
        ret_top.permute(0, 1, 3, 2, 4),
        ret_mid.permute(0, 1, 3, 2, 4),
        ret_bot.permute(0, 1, 3, 2, 4)
    )