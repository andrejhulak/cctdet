from ds import VisDrone
from torch.utils.data import DataLoader
import torch
from models.yolov11.yolo11 import Yolo11
from torchsummary import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
  root = "datasets/VisDrone/VisDrone2019-DET-val"
  ds = VisDrone(root=root)
  dl = DataLoader(dataset=ds,
                  batch_size=2)

  model = Yolo11(d=0.5, w=0.5, mc=1024).to(device)
  x = torch.zeros((1, 3, 640, 640)).to(device)
  # print(model(x).shape)
  summary(model, (3, 640, 640))

