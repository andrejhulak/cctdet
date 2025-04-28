import torch
import torchvision
from collections import OrderedDict
from torchvision.models.detection.faster_rcnn import RegionProposalNetwork, RPNHead
from torchvision.models.detection.faster_rcnn import _default_anchorgen, AnchorGenerator
from torchvision.models import MobileNet_V2_Weights
from torchvision.models.detection.transform import GeneralizedRCNNTransform

class CCTdeT(torch.nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    min_size=800
    max_size=1333
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, **kwargs)
 
    # output channels in a backbone, for mobilenet_v2, it's 1280
    self.backbone = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
    out_channels = 1280

    rpn_pre_nms_top_n_train=2000
    rpn_pre_nms_top_n_test=1000
    rpn_post_nms_top_n_train=2000
    rpn_post_nms_top_n_test=1000

    rpn_anchor_generator = AnchorGenerator(
      sizes=((32, 64, 128, 256, 512),),
      aspect_ratios=((0.5, 1.0, 2.0),)
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
                                     nms_thresh=0.7,
                                     score_thresh=0.0)

  def forward(self, images):
    images, targets = self.transform(images, None)
    features = self.backbone(images.tensors)

    features = self.backbone(images)
    targets = None

    if isinstance(features, torch.Tensor):
      features = OrderedDict([("0", features)])

    proposals, proposal_losses = self.rpn(images, features, targets)
    return proposals