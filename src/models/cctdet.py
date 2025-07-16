import torch
from torchvision.models.detection.faster_rcnn import RegionProposalNetwork, RPNHead, AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.roi_heads import RoIHeads
from vit_pytorch.cct import CCT
from utils.misc import class_names
from torchvision.ops import MultiScaleRoIAlign
from models.cctpredictor import CCTPredictor
from typing import List, Tuple
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor 
from torchvision.models.resnet import resnet50
from torchvision.models.resnet import ResNet50_Weights

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = len(class_names) + 1 # for the background class

class CCTdeT(torch.nn.Module):
  def __init__(self, model_config, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # standard Faster-RCNN transform 
    self.transform = GeneralizedRCNNTransform(
      min_size=800, max_size=1333,
      image_mean=[0.485, 0.456, 0.406],
      image_std=[0.229, 0.224, 0.225]
    )

    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2, progress=True)
    backbone = _resnet_fpn_extractor(backbone, 5, norm_layer=torch.nn.BatchNorm2d)
    self.backbone = backbone

    rpn_pre_nms_top_n_train=2000
    rpn_pre_nms_top_n_test=2000
    rpn_post_nms_top_n_train=1000
    rpn_post_nms_top_n_test=1000

    # anchor_sizes = ((4,), (8,), (16,), (32,), (64,))
    # anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
    anchor_sizes = ((12,), (18,), (24,), (64,), (128,))
    aspect_ratios = ((0.5, 1.0, 1.5),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    rpn_head = RPNHead(backbone.out_channels, anchor_generator.num_anchors_per_location()[0], conv_depth=2)

    rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
    rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

    self.rpn = RegionProposalNetwork(anchor_generator=anchor_generator,
                                    head=rpn_head,
                                    fg_iou_thresh=0.7,
                                    bg_iou_thresh=0.3,
                                    batch_size_per_image=512,
                                    positive_fraction=0.5,
                                    pre_nms_top_n=rpn_pre_nms_top_n,
                                    post_nms_top_n=rpn_post_nms_top_n,
                                    nms_thresh=0.5,
                                    score_thresh=0.0)

    dim = model_config['dim']
    box_output_size = model_config['box_output_size']
    box_roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3', '4'], output_size=box_output_size, sampling_ratio=2)

    cct = CCT(img_size=(box_output_size,box_output_size),
              embedding_dim=dim,
              n_input_channels=backbone.out_channels,
              n_conv_layers=model_config['n_conv_layers'],
              kernel_size=model_config['kernel_size'], stride=model_config['stride'], padding=model_config['padding'],
              pooling_kernel_size=model_config['pooling_kernel_size'], pooling_stride=model_config['pooling_stride'],
              pooling_padding=model_config['pooling_padding'],
              num_layers=model_config['num_layers'], num_heads=model_config['num_heads'], mlp_ratio=model_config['mlp_ratio'],
              num_classes=num_classes,
              positional_embedding='learnable')

    predictor = CCTPredictor(cct, embed_dim=dim, num_classes=num_classes)

    self.roi_heads = RoIHeads(box_roi_pool=box_roi_pool,
                              box_head=torch.nn.Identity(),
                              box_predictor=predictor,
                              fg_iou_thresh=0.5, bg_iou_thresh=0.5,
                              batch_size_per_image=512, positive_fraction=0.25,
                              bbox_reg_weights=None,
                              score_thresh=0.05, nms_thresh=0.5, detections_per_img=300)

  def forward(self, batch, **kwargs):
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
      images = batch
      targets = None
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

      return detections