import torch
from torch.optim import AdamW, SGD
from datasets.ds import VisDrone
from torch.utils.data import DataLoader
from utils.misc import collate_fn_simple, format_metrics, class_names, load_config_from_json
from models.cctdet import CCTdeT
from engine import evaluate, conf_mat
from collections import Counter, OrderedDict
from tqdm import tqdm
import numpy as np
import os
from ultralytics.utils.metrics import ConfusionMatrix
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4

if __name__ == "__main__":
  val_root = "data/VisDrone2019-DET-val"
  val_ds = VisDrone(root=val_root)
  val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                      collate_fn=collate_fn_simple)

  model_number = 9
  model_config = load_config_from_json(model_number)
  model = CCTdeT(model_config)
  # ckpt_path = "runs/detect/fasterrcnn3/best.pt"
  # ckpt_path = "runs/detect/wow3/last.pt"
  ckpt_path = f'runs/detect/train{model_number}/weights/best.pt'
  checkpoint = torch.load(ckpt_path, weights_only=False, map_location=device)

  if 'ema' in checkpoint and hasattr(checkpoint['ema'], 'state_dict'):
      ema_state_dict = checkpoint['ema'].state_dict()
  elif 'model' in checkpoint and hasattr(checkpoint['model'], 'state_dict'):
      ema_state_dict = checkpoint['model'].state_dict()
  else:
      raise KeyError("Could not find compatible model state_dict in checkpoint.")

  model.load_state_dict(ema_state_dict, strict=True)
  model.to(device).to(torch.float32)

  total_params = sum(p.numel() for p in model.parameters())
  print(total_params)

  metrics = evaluate(model, val_dl)
  print(format_metrics(metrics))

  # mat = conf_mat(model, val_dl)
  # mat.plot()