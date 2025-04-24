import torch.nn as nn
from models.yolov11.convblock import Conv

class PSABlock(nn.Module):
  def __init__(self, c, num_head):
    super().__init__()
    self.conv1 = Attention(c, num_head)
    self.conv2 = nn.Sequential(Conv(c, c * 2, p=0), Conv(c * 2, c, p=0))

  def forward(self, x):
    x = x + self.conv1(x)
    return x + self.conv2(x)

class Attention(nn.Module):
  def __init__(self, c, num_head):
    super().__init__()
    self.num_head = num_head
    self.dim_head = c // num_head
    self.dim_key = self.dim_head // 2
    self.scale = self.dim_key ** -0.5

    self.qkv = Conv(c, c + self.dim_key * num_head * 2, k=1, s=1, p=0, act=False)

    self.conv1 = Conv(c, c, k=3, s=1)
    self.conv2 = Conv(c, c, k=1, s=1, p=0)

  def forward(self, x):
    b, c, h, w = x.shape

    qkv = self.qkv(x)
    qkv = qkv.view(b, self.num_head, self.dim_key * 2 + self.dim_head, h * w)

    q, k, v = qkv.split([self.dim_key, self.dim_key, self.dim_head], dim=2)

    attn = (q.transpose(-2, -1) @ k) * self.scale
    attn = attn.softmax(dim=-1)

    x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.conv1(v.reshape(b, c, h, w))
    return self.conv2(x)