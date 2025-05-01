import torch
import torchvision
from collections import OrderedDict
from torchvision.models.detection.faster_rcnn import RegionProposalNetwork, RPNHead, AnchorGenerator
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from vit_pytorch.cct import CCT
from utils.misc import class_names
from models.roi_head_modifications import RoIHeads
from torchvision.ops import MultiScaleRoIAlign
from models.cctpredictor import CCTPredictor
from typing import Dict, List, Optional, Tuple, Union
from torchvision.models.detection.faster_rcnn import TwoMLPHead
import types
from models.rpn_modifications import compute_loss, forward

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = len(class_names)

class CCTdeT(torch.nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # TODO probaly remove this and do something of our own, makes no sense that it's here twice
    min_size=800
    max_size=1333
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
 
    # backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
    # self.backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    # out_channels = 2048

    self.backbone = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
    out_channels = 1280

    rpn_pre_nms_top_n_train=200
    rpn_pre_nms_top_n_test=100
    rpn_post_nms_top_n_train=200
    rpn_post_nms_top_n_test=100

    rpn_anchor_generator = AnchorGenerator(
      sizes=((32, 64, 128, 256, 512),),
      aspect_ratios=((1.0, 1.5, 2.0),)
    )

    rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

    rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
    rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

    self.rpn = RegionProposalNetwork(anchor_generator=rpn_anchor_generator,
                                    head=rpn_head,
                                    fg_iou_thresh=0.7,
                                    bg_iou_thresh=0.3,
                                    batch_size_per_image=256,
                                    positive_fraction=0.5,
                                    pre_nms_top_n=rpn_pre_nms_top_n,
                                    post_nms_top_n=rpn_post_nms_top_n,
                                    nms_thresh=0.5,
                                    score_thresh=0.0)

    box_output_size = 32
    box_roi_pool = MultiScaleRoIAlign(featmap_names=['0'], output_size=box_output_size, sampling_ratio=2)

    cct = CCT(img_size=(box_output_size,box_output_size),
              embedding_dim=128,
              n_input_channels=out_channels,
              n_conv_layers=2,
              kernel_size=3, stride=1, padding=1,
              pooling_kernel_size=2, pooling_stride=2,
              pooling_padding=0,
              num_layers=2, num_heads=2, mlp_ratio=2.0,
              num_classes=num_classes,
              positional_embedding='learnable')

    predictor = CCTPredictor(cct, embed_dim=128, num_classes=num_classes)

    self.roi_heads = RoIHeads(box_roi_pool=box_roi_pool,
                              box_head=torch.nn.Identity(),
                              box_predictor=predictor,
                              fg_iou_thresh=0.5, bg_iou_thresh=0.5,
                              batch_size_per_image=32, positive_fraction=0.25,
                              bbox_reg_weights=None,
                              score_thresh=0.05, nms_thresh=0.5, detections_per_img=200)
    # loss changes
    self.rpn.compute_loss = types.MethodType(compute_loss, self.rpn)
    self.rpn.forward = types.MethodType(forward, self.rpn)

  def forward(self, images, targets=None):
    if self.training:
      if targets is None:
        torch._assert(False, "targets should not be none when in training mode")
      else:
        for target in targets:
          boxes = target["boxes"]
          if isinstance(boxes, torch.Tensor):
            torch._assert(
              len(boxes.shape) == 2 and boxes.shape[-1] == 4,
              f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
            )
          else:
            torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
      val = img.shape[-2:]
      torch._assert(
        len(val) == 2,
        f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
      )
      original_image_sizes.append((val[0], val[1]))

    images, targets = self.transform(images, targets)
    images = images.to(device)

    features = self.backbone(images.tensors)

    if isinstance(features, torch.Tensor):
      features = OrderedDict([("0", features)])

    proposals, proposal_losses = self.rpn(images, features, targets)
    detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
    detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

    losses = {}
    losses.update(detector_losses)
    losses.update(proposal_losses)

    if self.training:
      return losses
    else:
      return detections