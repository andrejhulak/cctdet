import torch
import torchvision
from collections import OrderedDict
from torchvision.models.detection.faster_rcnn import RegionProposalNetwork, RPNHead, AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.roi_heads import RoIHeads
from vit_pytorch.cct import CCT
from utils.misc import class_names
# from models.roi_head_modifications import RoIHeads
from torchvision.ops import MultiScaleRoIAlign
from models.cctpredictor import CCTPredictor
from typing import Dict, List, Optional, Tuple, Union
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = len(class_names)

class CCTdeT(torch.nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.transform = GeneralizedRCNNTransform(
      min_size=640, max_size=640,
      image_mean=[0.485, 0.456, 0.406],
      image_std=[0.229, 0.224, 0.225]
    )
 
    backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=5)
    self.backbone = backbone

    rpn_pre_nms_top_n_train=2000
    rpn_pre_nms_top_n_test=1000
    rpn_post_nms_top_n_train=1000
    rpn_post_nms_top_n_test=1000

    anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    rpn_head = RPNHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0])

    rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
    rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

    self.rpn = RegionProposalNetwork(anchor_generator=anchor_generator,
                                    head=rpn_head,
                                    fg_iou_thresh=0.7,
                                    bg_iou_thresh=0.3,
                                    batch_size_per_image=256,
                                    positive_fraction=0.5,
                                    pre_nms_top_n=rpn_pre_nms_top_n,
                                    post_nms_top_n=rpn_post_nms_top_n,
                                    nms_thresh=0.5,
                                    score_thresh=0.0)

    box_output_size = 28
    box_roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=box_output_size, sampling_ratio=2)

    cct = CCT(img_size=(box_output_size,box_output_size),
              embedding_dim=512,
              n_input_channels=backbone.out_channels,
              n_conv_layers=2,
              kernel_size=3, stride=1, padding=1,
              pooling_kernel_size=2, pooling_stride=2,
              pooling_padding=0,
              num_layers=2, num_heads=2, mlp_ratio=2.0,
              num_classes=num_classes,
              positional_embedding='learnable')

    predictor = CCTPredictor(cct, embed_dim=512, num_classes=num_classes)

    self.roi_heads = RoIHeads(box_roi_pool=box_roi_pool,
                              box_head=torch.nn.Identity(),
                              box_predictor=predictor,
                              fg_iou_thresh=0.5, bg_iou_thresh=0.5,
                              batch_size_per_image=256, positive_fraction=0.25,
                              bbox_reg_weights=None,
                              score_thresh=0.05, nms_thresh=0.5, detections_per_img=200)

  def forward(self, batch, **kwargs):
    if isinstance(batch, dict):
      for name, param in self.named_parameters():
        dtype = param.data.dtype
        break

      images = [img.to(dtype) / 255.0 for img in batch['img']]
      batch_idx = batch['batch_idx']
      boxes_all = batch['bboxes']
      labels_all = batch['cls'].view(-1).to(torch.int64)

      targets = []
      for i in range(len(images)):
        mask = (batch_idx == i)
        boxes_normalized_xywh = boxes_all[mask]
        labels = labels_all[mask]

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
        
      images = [img.to(device) for img in images]
      for t in targets:
        t['boxes'] = t['boxes'].to(device)
        t['labels'] = t['labels'].to(device)

      original_image_sizes: List[Tuple[int, int]] = []
      for img in images:
        val = img.shape[-2:]
        torch._assert(
          len(val) == 2,
          f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
        )
        original_image_sizes.append((val[0], val[1]))

      images, targets = self.transform(images, targets)
      features = self.backbone(images.tensors)

      proposals, proposal_losses = self.rpn(images, features, targets)
      detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
      detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

      losses = {}
      losses.update(detector_losses)
      losses.update(proposal_losses)

      if self.training:
        total_loss = sum(losses.values())
        loss_items = torch.stack([losses[k] for k in losses.keys()])
        return total_loss, loss_items
      else:
        return detections
    else: 
      images = [img.to(torch.float32) / 255.0 for img in batch]
      images = [img.to(device) for img in images]
      original_image_sizes = [img.shape[-2:] for img in images]

      images, _ = self.transform(images, None)
      features = self.backbone(images.tensors)

      proposals, _ = self.rpn(images, features, None)
      detections, _ = self.roi_heads(features, proposals, images.image_sizes, None)
      detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

      return detections

  def loss(self, batch, preds):
    if self.training:
      return self.forward(batch)
    else:
      return torch.tensor(0.0, device=device), torch.zeros(4, device=device)