import torch
import torchvision
from collections import OrderedDict
from torchvision.models.detection.faster_rcnn import RegionProposalNetwork, RPNHead, AnchorGenerator
from torchvision.models import MobileNet_V2_Weights
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from vit_pytorch.cct import CCT
from utils.misc import class_names
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import MultiScaleRoIAlign
from models.cctpredictor import CCTPredictor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = len(class_names)

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

    cct = CCT(img_size=(7,7),
              embedding_dim=256,
              n_input_channels=out_channels,
              n_conv_layers=2,
              kernel_size=3, stride=1, padding=1,
              pooling_kernel_size=2, pooling_stride=2,
              pooling_padding=0,
              num_layers=4, num_heads=4, mlp_ratio=2.0,
              num_classes=num_classes,
              positional_embedding='learnable')

    box_roi_pool = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

    predictor = CCTPredictor(cct, embed_dim=256, num_classes=num_classes)

    self.roi_heads = RoIHeads(box_roi_pool=box_roi_pool,
                              box_head=torch.nn.Identity(),
                              box_predictor=predictor,
                              fg_iou_thresh=0.5, bg_iou_thresh=0.5,
                              batch_size_per_image=512, positive_fraction=0.25,
                              bbox_reg_weights=None,
                              score_thresh=0.05, nms_thresh=0.5, detections_per_img=100)

  def forward(self, images, targets=None):
    images, targets = self.transform(images, targets)
    images = images.to(device)
    features = self.backbone(images.tensors)

    if isinstance(features, torch.Tensor):
      features = OrderedDict([("0", features)])

    proposals, proposal_losses = self.rpn(images, features, targets)
    detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
    return detections