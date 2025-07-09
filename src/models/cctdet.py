import torch
import torchvision
from collections import OrderedDict
from torchvision.models.detection.faster_rcnn import RegionProposalNetwork, RPNHead, AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.roi_heads import RoIHeads
from vit_pytorch.cct import CCT
from utils.misc import class_names
from torchvision.ops import MultiScaleRoIAlign
from models.cctpredictor import CCTPredictor
from typing import Dict, List, Optional, Tuple, Union
from torchvision.models.detection.faster_rcnn import FastRCNNConvFCHead, fasterrcnn_resnet50_fpn_v2, FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _resnet_fpn_extractor 
from torchvision.models.resnet import resnet50
from torchvision.models.resnet import ResNet50_Weights
from utils.misc import load_part_of_pretrained_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = len(class_names) + 1 # for the background class

class CCTdeT(torch.nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.transform = GeneralizedRCNNTransform(
      min_size=800, max_size=1333,
      image_mean=[0.485, 0.456, 0.406],
      image_std=[0.229, 0.224, 0.225]
    )
 
    # backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=5)
    # self.backbone = backbone

    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2, progress=True)
    backbone = _resnet_fpn_extractor(backbone, 5, norm_layer=torch.nn.BatchNorm2d)
    self.backbone = backbone

    rpn_pre_nms_top_n_train=2000
    rpn_pre_nms_top_n_test=2000
    rpn_post_nms_top_n_train=1000
    rpn_post_nms_top_n_test=1000

    # anchor_sizes = ((4,), (8,), (16,), (32,), (64,))
    anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
    aspect_ratios = ((0.5, 1.0, 1.5),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    rpn_head = RPNHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], conv_depth=2)

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

    # box_output_size = 7

    # box_roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3', '4'], output_size=box_output_size, sampling_ratio=2)

    # box_head = FastRCNNConvFCHead(
    #     (backbone.out_channels, box_output_size, box_output_size), [256, 256, 256, 256], [1024], norm_layer=torch.nn.BatchNorm2d
    # )

    # representation_size = 1024
    # predictor = FastRCNNPredictor(representation_size, num_classes)

    # self.roi_heads = RoIHeads(box_roi_pool=box_roi_pool,
    #                           box_head=box_head,
    #                           box_predictor=predictor,
    #                           fg_iou_thresh=0.5, bg_iou_thresh=0.5,
    #                           batch_size_per_image=256, positive_fraction=0.25,
    #                           bbox_reg_weights=None,
    #                           score_thresh=0.05, nms_thresh=0.5, detections_per_img=300)

    dim = 384
    box_output_size = 7
    box_roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3', '4'], output_size=box_output_size, sampling_ratio=2)

    cct = CCT(img_size=(box_output_size,box_output_size),
              embedding_dim=dim,
              n_input_channels=backbone.out_channels,
              n_conv_layers=2,
              kernel_size=7, stride=2, padding=3,
              pooling_kernel_size=3, pooling_stride=2,
              pooling_padding=1,
              num_layers=4, num_heads=4, mlp_ratio=3.0,
              num_classes=num_classes,
              positional_embedding='learnable')

    predictor = CCTPredictor(cct, embed_dim=dim, num_classes=num_classes)

    self.roi_heads = RoIHeads(box_roi_pool=box_roi_pool,
                              box_head=torch.nn.Identity(),
                              box_predictor=predictor,
                              fg_iou_thresh=0.5, bg_iou_thresh=0.5,
                              batch_size_per_image=256, positive_fraction=0.25,
                              bbox_reg_weights=None,
                              score_thresh=0.05, nms_thresh=0.5, detections_per_img=300)

  def forward(self, batch, **kwargs):
    # do we even need this?
    if isinstance(batch, dict):
      images = batch['images']
      targets = batch['targets']
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
        loss_items = torch.stack([losses[k] for k in losses.keys()]).detach().cpu()
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