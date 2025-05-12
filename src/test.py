from ultralytics.data.dataset import YOLODataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.engine.validator import BaseValidator
from torch.utils.data import DataLoader
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
from models.cctdet import CCTdeT
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.detect import DetectionTrainer
from copy import copy
import torch
from torchmetrics.detection import MeanAveragePrecision
from ultralytics.utils import LOGGER
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from utils.misc import load_part_of_pretrained_model

class CCTValidator(DetectionValidator):
  def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
    super().__init__(dataloader, save_dir, pbar, args, _callbacks)
    self.args.task = "detect"
    self.metrics = MeanAveragePrecision()
    self.metrics.keys = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    self.seen = 0
    self.nt_per_class = torch.zeros(10, dtype=torch.int64, device='cuda')
    self.stats = dict()

  # def preprocess(self, batch):
  #   return batch

  # def postprocess(self, preds):
  #   results = []
  #   for det in preds:
  #     if len(det["boxes"]) == 0:
  #       results.append(torch.zeros((0, 6), device=det["boxes"].device))
  #       continue
  #     boxes = det["boxes"]
  #     scores = det["scores"].unsqueeze(1)
  #     labels = det["labels"].unsqueeze(1).to(torch.float)
  #     results.append(torch.cat([boxes, scores, labels], dim=1))
  #   return results

  # def update_metrics(self, preds, batch):
  #   metric_preds = []
  #   metric_targets = []

  #   for i, det in enumerate(preds):
  #     pred_boxes = det[:, :4].cpu()
  #     pred_scores = det[:, 4].cpu()
  #     pred_labels = det[:, 5].cpu().int()
  #     metric_preds.append(dict(boxes=pred_boxes, scores=pred_scores, labels=pred_labels))

  #     img_idx = batch["batch_idx"] == i
  #     gt_boxes_normalized = batch["bboxes"][img_idx].clone()
  #     gt_labels_ = batch["cls"][img_idx].cpu().int().flatten()

  #     img_h, img_w = batch["img"][i].shape[1:]
  #     x_center = gt_boxes_normalized[:, 0] * img_w
  #     y_center = gt_boxes_normalized[:, 1] * img_h
  #     width = gt_boxes_normalized[:, 2] * img_w
  #     height = gt_boxes_normalized[:, 3] * img_h
  #     x1 = x_center - width / 2
  #     y1 = y_center - height / 2
  #     x2 = x_center + width / 2
  #     y2 = y_center + height / 2
  #     gt_boxes_abs = torch.stack([x1, y1, x2, y2], dim=1).cpu()

  #     metric_targets.append(dict(boxes=gt_boxes_abs, labels=gt_labels_))

  #   self.metrics.update(metric_preds, metric_targets)

  #   metric_results = self.metrics.compute()

  #   self.seen += len(batch["img"])
    
  #   for targets in metric_targets:
  #     labels = targets["labels"]
  #     for lbl in labels:
  #       self.nt_per_class[lbl] += 1

  #   self.metrics.map = metric_results["map"].item()
  #   self.metrics.map50 = metric_results["map_50"].item()
  #   self.metrics.map75 = metric_results["map_75"].item()
  #   if "map_per_class" in metric_results:
  #     self.metrics.maps = metric_results["map_per_class"].cpu().tolist()
  #   else:
  #     self.metrics.maps = [metric_results["map"].item()] * len(self.data["names"])

  # def get_desc(self):
  #   return ("%22s" + "%11s" * 4) % (
  #       "Class", "Images", "Instances", "Precision", "Recall"
  #   )

  # def get_stats(self):
  #   return {
  #     "metrics/precision(B)": self.metrics.map50,
  #     "metrics/recall(B)": self.metrics.map50,
  #     "metrics/mAP50(B)": self.metrics.map50,
  #     "metrics/mAP50-95(B)": self.metrics.map,
  #   }
        
  # def reset_metrics(self):
  #   self.metrics.reset()
  #   self.seen = 0
  #   self.nt_per_class = torch.zeros(len(self.data["names"]), dtype=torch.int64)

  # def print_results(self):
  #   stats = self.get_stats()
  #   pf = "%22s" + "%11.3g" * 4
  #   LOGGER.info(pf % ("all", 
  #                     stats["metrics/precision(B)"],
  #                     stats["metrics/recall(B)"],
  #                     stats["metrics/mAP50(B)"],
  #                     stats["metrics/mAP50-95(B)"]))

class CCTTrainer(BaseTrainer):
  def get_model(self, cfg=None, weights=None, verbose=None):
    fasterrcnn_state_dict = fasterrcnn_resnet50_fpn_v2(FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1).state_dict()
    model = CCTdeT()
    model = load_part_of_pretrained_model(fasterrcnn_state_dict, model)

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
    model.to(torch.float32)
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
    self.loss_names = ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")
    return CCTValidator(
        self.test_loader,
        save_dir=self.save_dir,
        args=copy(self.args),
        _callbacks=self.callbacks
    )

if __name__ == "__main__":
  overrides = {
    'data': 'C:/Users/User/Desktop/andrej/cctdet/src/VisDrone.yaml',
    'epochs': 30,
    'batch': 1,
    'device': '0'
  }

  trainer = CCTTrainer(overrides=overrides)
  trainer.train()