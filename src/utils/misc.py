import torch

class_names = {
  0: 'pedestrian',
  1: 'people',
  2: 'bicycle',
  3: 'car',
  4: 'van',
  5: 'truck',
  6: 'tricycle',
  7: 'awning-tricycle',
  8: 'bus',
  9: 'motor'
}

def collate_fn(batch):
  # ret = {}
  imgs = [t[0] for t in batch] 
  # labels = [t[1] for t in batch]

  # for item in labels:
  #   print(item['boxes'].shape)
  #   print(item['labels'].shape)

  lengths_imgs = torch.tensor([t.shape[0] for t in imgs])
  batch_imgs = ([t for t in imgs])
  mask_imgs = (batch_imgs != 0)
  # ret['imgs'] = batch_imgs, lengths_imgs, mask_imgs

  # lengths_labels = torch.tensor([t.shape[0] for t in labels])
  # batch_labels = pad_sequence([t for t in labels])
  # mask_labels = (batch_labels != 0)
  # ret['labels'] = batch_labels, lengths_labels, mask_labels

  return batch_imgs, lengths_imgs, mask_imgs

def collate_fn_simple(batch):
  images = []
  targets = []
  for image, target in batch:
    images.append(image)
    targets.append(target)
  return images, targets

def format_metrics(metrics, class_names=class_names):
  formatted_output = "Evaluation Metrics:\n"
  formatted_output += f"  Mean Average Precision (mAP): {metrics['map']:.4f}\n"
  formatted_output += f"  mAP@0.5 IoU: {metrics['map_50']:.4f}\n"
  formatted_output += f"  mAP@0.75 IoU: {metrics['map_75']:.4f}\n"
  formatted_output += f"  mAP Small Objects: {metrics['map_small']:.4f}\n"
  formatted_output += f"  mAP Medium Objects: {metrics['map_medium']:.4f}\n"
  formatted_output += f"  mAP Large Objects: {metrics['map_large']:.4f}\n"
  formatted_output += f"  Mean Average Recall (mAR) @ 1: {metrics['mar_1']:.4f}\n"
  formatted_output += f"  mAR @ 10: {metrics['mar_10']:.4f}\n"
  formatted_output += f"  mAR @ 100: {metrics['mar_100']:.4f}\n"
  formatted_output += f"  mAR Small Objects: {metrics['mar_small']:.4f}\n"
  formatted_output += f"  mAR Medium Objects: {metrics['mar_medium']:.4f}\n"
  formatted_output += f"  mAR Large Objects: {metrics['mar_large']:.4f}\n"

  if 'map_per_class' in metrics and 'classes' in metrics:
    formatted_output += "\n  mAP per Class:\n"
    for i, value in enumerate(metrics['map_per_class']):
      class_index = metrics['classes'][i].item()
      if class_index in class_names:
        formatted_output += f"    {class_names[class_index]}: {value:.4f}\n"
      else:
        formatted_output += f"    Class {class_index}: {value:.4f}\n"

  if 'mar_100_per_class' in metrics and 'classes' in metrics:
    formatted_output += "\n  mAR@100 per Class:\n"
    for i, value in enumerate(metrics['mar_100_per_class']):
      class_index = metrics['classes'][i].item()
      if class_index in class_names:
        formatted_output += f"    {class_names[class_index]}: {value:.4f}\n"
      else:
        formatted_output += f"    Class {class_index}: {value:.4f}\n"

  return formatted_output