from ultralytics.data.dataset import YOLODataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.engine.validator import BaseValidator
from torch.utils.data import DataLoader
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
from models.cctdet import CCTdeT
from ultralytics.models.yolo.detect import DetectionValidator
from copy import copy
import torch
from ultralytics.utils import LOGGER

class CCTValidator(DetectionValidator):
  def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
    super().__init__(dataloader, save_dir, pbar, args)
    self.args.task = "detect"

class CCTTrainer(BaseTrainer):
  def get_model(self, cfg=None, weights=None, verbose=None):
    model = CCTdeT()

    # ckpt_path = "runs/detect/train2/weights/best.pt"
    # checkpoint = torch.load(ckpt_path, weights_only=False, map_location=self.device)

    # if 'ema' in checkpoint and hasattr(checkpoint['ema'], 'state_dict'):
    #    ema_state_dict = checkpoint['ema'].state_dict()
    # elif 'ema_state_dict' in checkpoint:
    #    ema_state_dict = checkpoint['ema_state_dict']
    # elif 'model' in checkpoint:
    #    ema_state_dict = checkpoint['model'].state_dict()
    # else:
    #    raise KeyError("Could not find compatible model state_dict in checkpoint.")

    # from ultralytics.utils.torch_utils import de_parallel
    # ema_state_dict = de_parallel(ema_state_dict)

    # model.load_state_dict(ema_state_dict, strict=True)

    return model

  def build_dataset(self, img_path, mode="train", batch=None):
    dataset = build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val")
    return dataset

  def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
    assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
    with torch_distributed_zero_first(rank):
      dataset = self.build_dataset(dataset_path, mode, batch_size)
    shuffle = mode == "train"
    workers = self.args.workers if mode == "train" else self.args.workers * 3
    return build_dataloader(dataset, batch_size, workers, shuffle, rank) 

  def get_validator(self):
    self.loss_names = "cls_loss", "box_reg", "obj_loss", "rpn_loss"
    return CCTValidator(
        self.test_loader,
        save_dir=self.save_dir,
        args=copy(self.args),
        _callbacks=self.callbacks
    )

  def preprocess_batch(self, batch):
    device = self.device
    
    images = [img.to(device) / 255.0 for img in batch['img']]
    batch_idx = batch['batch_idx']
    boxes_all = batch['bboxes']
    labels_all = batch['cls'].view(-1).to(torch.int64)

    # prepare targets
    targets = []

    for i in range(len(images)):
      mask = (batch_idx == i)
      boxes_normalized_xywh = boxes_all[mask]
      # + 1 because of Faster R-CNN PyTorch code which ignores labels 0 because it thinks it's background
      labels = labels_all[mask] + 1

      original_h, original_w = images[i].shape[1:]

      boxes_unnormalized_xyxy = boxes_normalized_xywh.clone()

      center_x_norm = boxes_unnormalized_xyxy[:, 0]
      center_y_norm = boxes_unnormalized_xyxy[:, 1]
      width_norm = boxes_unnormalized_xyxy[:, 2]
      height_norm = boxes_unnormalized_xyxy[:, 3]

      x_min = (center_x_norm - width_norm / 2) * original_w
      y_min = (center_y_norm - height_norm / 2) * original_h
      x_max = (center_x_norm + width_norm / 2) * original_w
      y_max = (center_y_norm + height_norm / 2) * original_h

      boxes_unnormalized_xyxy[:, 0] = x_min
      boxes_unnormalized_xyxy[:, 1] = y_min
      boxes_unnormalized_xyxy[:, 2] = x_max
      boxes_unnormalized_xyxy[:, 3] = y_max

      targets.append({'boxes': boxes_unnormalized_xyxy, 'labels': labels})

      # makes sure they're one the correct device
      for t in targets:
        t['boxes'] = t['boxes'].to(device)
        t['labels'] = t['labels'].to(device)
    
    return {'images' : images, 'targets' : targets}

  def progress_string(self):
      return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
          "Epoch",
          "GPU_mem",
          *self.loss_names,
          "Instances",
          "Size",
      )

if __name__ == "__main__":
  overrides = {
    'data': 'data/VisDrone.yaml',
    'epochs': 30,
    'batch': 2,
    'device': '0',
    'imgsz' : 1920,
    'deterministic' : False
  }

  trainer = CCTTrainer(overrides=overrides)
  trainer.train()