import torch
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
from ultralytics.utils.metrics import ConfusionMatrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.no_grad()
def evaluate(model, dataloader):
  model.eval()
  # metric = MeanAveragePrecision(box_format='xyxy', iou_thresholds=[0.5], class_metrics=True).to(device)
  metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True).to(device)
  # :D this isn't a good solution but for now it's okay, we need to check how ultralytics does this evaluation
  metric.warn_on_many_detections = False
  
  for images, targets, _ in tqdm(dataloader):
    imgs = [img.to(device) for img in images]
    
    detections = model(imgs)
    
    preds = []
    for i, det in enumerate(detections):
      preds.append({
        'boxes': det['boxes'].detach().cpu(),
        'scores': det['scores'].detach().cpu(),
        'labels': det['labels'].detach().cpu().int()
      })

    gt_boxes = [t['boxes'].cpu() for t in targets]
    gt_labels = [t['labels'].flatten().cpu().int() for t in targets]

    gt = [{'boxes': b, 'labels': l} for b, l in zip(gt_boxes, gt_labels)]

    metric.update(preds, gt)
  
  return metric.compute()

@torch.no_grad()
def conf_mat(model, dataloader):
  model.eval()
  metric = ConfusionMatrix(nc=10)

  for images, targets, _ in tqdm(dataloader):
    imgs = [img.to(device) for img in images]
    detections = model(imgs)

    for i in range(len(imgs)):
      det = detections[i]

      boxes = det['boxes']
      scores = det['scores'].unsqueeze(1)
      labels = det['labels'].unsqueeze(1).int() - 1

      pred_tensor = torch.cat([boxes, scores, labels], dim=1).cpu()

      tgt = targets[i]
      gt_boxes = tgt['boxes'].cpu()
      gt_labels = tgt['labels'].unsqueeze(1).int().cpu() - 1

      metric.process_batch(pred_tensor, gt_boxes, gt_labels)

  return metric