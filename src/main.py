from datasets.ds import VisDrone
from torch.utils.data import DataLoader
import torch
from models.dummy import Dummy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
  root = "data/VisDrone/VisDrone2019-DET-val"
  ds = VisDrone(root=root)
  dl = DataLoader(dataset=ds,
                  batch_size=2)

  model = Dummy()
  x = torch.rand((4, 3, 640, 640))
  print(x.shape)