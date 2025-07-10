import torch
import json

class_names = {
  1: 'pedestrian',
  2: 'people',
  3: 'bicycle',
  4: 'car',
  5: 'van',
  6: 'truck',
  7: 'tricycle',
  8: 'ajning-tricycle',
  9: 'bus',
  10: 'motor'
}

def collate_fn_simple(batch):
  images = []
  targets = []
  img_paths = []
  for image, target, img_path in batch:
    images.append(image)
    targets.append(target)
    img_paths.append(img_path)
  return images, targets, img_paths

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

def load_part_of_pretrained_model(pretrained_dict, model):
  model_dict = model.state_dict()
  for k, v in pretrained_dict.items():
    if k in model_dict:
      pretrained_dict = {k: v}

  model_dict.update(pretrained_dict) 
  model.load_state_dict(model_dict)
  return model

def load_config_from_json(model_number):
  model_path = f'runs/detect/train{model_number}weights'
  with open(model_path):
    model_config = json.load(f'{model_path}/model_config.json')
  return model_config