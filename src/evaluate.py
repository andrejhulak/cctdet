import torch
from datasets.ds import VisDrone
from torch.utils.data import DataLoader
from utils.misc import collate_fn_simple, format_metrics, load_config_from_args
from models.cctdet import CCTdeT
from models.fasterrcnn import FasterRCNN
from engine import evaluate, conf_mat

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4

if __name__ == "__main__":
  val_root = "data/VisDrone2019-DET-val"
  val_ds = VisDrone(root=val_root)
  val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                      collate_fn=collate_fn_simple)

  # newly trained models setup 
  model_number = "7"
  model_config = load_config_from_args(model_number)
  print("Running eval with this model config:")
  print(model_config)
  model = CCTdeT(model_config)
  ckpt_path = f'runs/detect/train{model_number}/weights/last.pt'

  # best CCTdeT model setup
  # model_config = {
  #   'dim': 384,
  #   'box_output_size': 7,
  #   'n_conv_layers': 2,
  #   'kernel_size': 7,
  #   'stride': 2,
  #   'padding': 3,
  #   'pooling_kernel_size': 3,
  #   'pooling_stride': 2,
  #   'pooling_padding': 1,
  #   'num_layers': 4,
  #   'num_heads': 4,
  #   'mlp_ratio': 3.0
  # }
  # ckpt_path = "old_models/wow3/best.pt"

  # for testing inference speed with different configs
  # model_config = {
  #   'dim': 384,
  #   'box_output_size': 7,
  #   'n_conv_layers': 2,
  #   'kernel_size': 7,
  #   'stride': 1,
  #   'padding': 3,
  #   'pooling_kernel_size': 3,
  #   'pooling_stride': 1,
  #   'pooling_padding': 1,
  #   'num_layers': 2,
  #   'num_heads': 4,
  #   'mlp_ratio': 2.0
  # }
  # model = CCTdeT(model_config)

  # Faster R-CNN best model setup
  # model = FasterRCNN()
  # ckpt_path = "old_models/fasterrcnn3/best.pt"

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