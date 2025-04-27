from datasets.ds import VisDrone
from torch.utils.data import DataLoader
import torch
from models.dummy import DummyRandom
from utils.misc import collate_fn_simple, format_metrics
from engine import evaluate

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
  root = "data/VisDrone/VisDrone2019-DET-val"
  ds = VisDrone(root=root)
  dl = DataLoader(dataset=ds,
                  batch_size=4,
                  shuffle=True,
                  collate_fn=collate_fn_simple)
  
  model = DummyRandom()
  result = evaluate(model, dl)
  print(format_metrics(result))