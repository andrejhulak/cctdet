import torch
from torch.optim import AdamW, SGD
from datasets.ds import VisDrone
from torch.utils.data import DataLoader
from utils.misc import collate_fn_simple, format_metrics, class_names
from models.cctdet import CCTdeT
from engine import evaluate, train
from transforms import get_transforms
from collections import Counter, OrderedDict
from tqdm import tqdm
import numpy as np
import os
import cv2
from ultralytics.nn.tasks import attempt_load_weights

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1
CONF_THRESHOLD = 0.5
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 0, 255)

def visualize_predictions(model, dataloader, class_names, confidence_threshold=0.75):
    model.eval()
    with torch.no_grad():
      for images, targets, image_paths in tqdm(dataloader, desc="Visualizing Predictions"):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for i, output in enumerate(outputs):
          if output is not None and len(output) > 0:
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy().astype(int)

            image = cv2.imread(image_paths[i])
            h, w = image.shape[:2]

            for box, score, label in zip(boxes, scores, labels):
              if score > confidence_threshold:
                x1, y1, x2, y2 = map(int, box)
                class_name = class_names[label]
                confidence_text = f"{score:.2f}"
                label_text = f"{class_name} {confidence_text}"

                cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, 2)
                cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 2)

            cv2.imshow(f"Prediction {os.path.basename(image_paths[i])}", image)
            cv2.waitKey(5000)
        cv2.destroyAllWindows()

if __name__ == "__main__":
  val_root = "data/VisDrone/VisDrone2019-DET-val"
  val_ds = VisDrone(root=val_root, transforms=None)
  val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                      collate_fn=collate_fn_simple, num_workers=4)

  model = CCTdeT()
  ckpt_path = "runs/detect/biggest/best.pt"
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
  print(f"Total parameters: {total_params}")

  visualize_predictions(model, val_dl, class_names, confidence_threshold=CONF_THRESHOLD)