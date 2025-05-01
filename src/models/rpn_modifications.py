import torch
from torch import Tensor
from torchvision.ops.ciou_loss import complete_box_iou_loss
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.rpn import concat_box_prediction_layers

def compute_loss(
  self, objectness: Tensor, pred_bbox_deltas: Tensor, labels: List[Tensor], regression_targets: List[Tensor], anchors: List[Tensor]
) -> Tuple[Tensor, Tensor]:
  sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
  sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
  sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

  sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

  objectness = objectness.flatten()

  labels = torch.cat(labels, dim=0)
  regression_targets = torch.cat(regression_targets, dim=0)

  pred_boxes = self.box_coder.decode(pred_bbox_deltas, anchors)
  target_boxes = self.box_coder.decode(regression_targets, anchors)

  box_loss = complete_box_iou_loss(
    boxes1=pred_boxes[sampled_pos_inds],
    boxes2=target_boxes[sampled_pos_inds],
    reduction='sum',
    eps=1e-7
  ) / (sampled_inds.numel())

  objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds])

  return objectness_loss, box_loss

def forward(
  self,
  images: ImageList,
  features: Dict[str, Tensor],
  targets: Optional[List[Dict[str, Tensor]]] = None,
) -> Tuple[List[Tensor], Dict[str, Tensor]]:
  features = list(features.values())
  objectness, pred_bbox_deltas = self.head(features)
  anchors = self.anchor_generator(images, features)

  num_images = len(anchors)
  num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
  num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
  objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)

  proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
  proposals = proposals.view(num_images, -1, 4)
  boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

  losses = {}
  if self.training:
    if targets is None:
      raise ValueError("targets should not be None")
    labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
    regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
    loss_objectness, loss_rpn_box_reg = self.compute_loss(
      objectness, pred_bbox_deltas, labels, regression_targets, anchors
    )
    losses = {
      "loss_objectness": loss_objectness,
      "loss_rpn_box_reg": loss_rpn_box_reg,
    }
  return boxes, losses