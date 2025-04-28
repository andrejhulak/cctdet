from datasets.ds import VisDrone
from torch.utils.data import DataLoader
import torch
from models.dummy import DummyRandom
from models.cctdet import CCTdeT
from utils.misc import collate_fn_simple, format_metrics
from engine import evaluate, visualize_image
from torchsummary import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
  root = "data/VisDrone/VisDrone2019-DET-val"
  ds = VisDrone(root=root)
  dl = DataLoader(dataset=ds,
                  batch_size=4,
                  shuffle=True,
                  collate_fn=collate_fn_simple)
  
  model_cct = CCTdeT().to(device)
  model_cct.eval()

  metrics = evaluate(model_cct, dl)
  print(format_metrics(metrics))