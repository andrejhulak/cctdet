import torch
from torch.optim import AdamW
from datasets.ds import VisDrone
from torch.utils.data import DataLoader
from utils.misc import collate_fn_simple
from models.cctdet import CCTdeT
from engine import evaluate, train
from utils.misc import format_metrics
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2
from utils.misc import class_names
from transforms import get_transforms 
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16

if __name__ == "__main__":
  train_root = "data/VisDrone/VisDrone2019-DET-train"
  train_ds = VisDrone(root=train_root, transforms=get_transforms())
  train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                        collate_fn=collate_fn_simple)

  model = ssdlite320_mobilenet_v3_large(num_classes=len(class_names)).to(device)
  # model = CCTdeT().to(device)
  # model = fasterrcnn_resnet50_fpn_v2(num_classes=len(class_names)).to(device)
  # checkpoint = torch.load("model_weights/cctdet/weights/model_epoch_1.pth")
  # model.load_state_dict(checkpoint)
  
  optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.0005)

  total_params = sum(p.numel() for p in model.parameters())
  print(total_params)

  epochs = 2
  for epoch in range(1, epochs+1):
    avg_loss = train(model, train_dl, optimizer, epoch)
    print(f"Epoch {epoch}/{epochs}  loss: {avg_loss:.4f}")

  val_root = "data/VisDrone/VisDrone2019-DET-val"
  val_ds = VisDrone(root=val_root)
  val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                      collate_fn=collate_fn_simple)
  metrics = evaluate(model, val_dl)
  print(format_metrics(metrics))