import torch
import torchvision
from collections import OrderedDict
from torchvision.models.detection.faster_rcnn import RegionProposalNetwork, RPNHead
from torchvision.models.detection.faster_rcnn import _default_anchorgen
from torchvision.models import MobileNet_V2_Weights

class CCTdeT(torch.nn.Module):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
 
    # output channels in a backbone. For mobilenet_v2, it's 1280
    self.backbone = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
    out_channels = 1280

    rpn_pre_nms_top_n_train=2000,
    rpn_pre_nms_top_n_test=1000,
    rpn_post_nms_top_n_train=2000,
    rpn_post_nms_top_n_test=1000,

    rpn_anchor_generator = _default_anchorgen()
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

  def forward(self, x):
    features = self.backbone(x)

    if isinstance(features, torch.Tensor):
      features = OrderedDict([("0", features)])

    proposals, proposal_losses = self.rpn(x, features)
    return proposals