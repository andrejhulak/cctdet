from itertools import product
from cct_trainer import CCTTrainer

def train_model(model_config):
  overrides = {
    'data': 'data/VisDrone.yaml',
    'epochs': 2,
    'batch': 2,
    'device': '0',
    'imgsz' : 1920,
    'deterministic' : False,
    'model_config' : model_config
  }

  trainer = CCTTrainer(overrides=overrides)
  trainer.train()

if __name__ == "__main__":
  dims = [32, 64, 128, 256]
  box_output_sizes = [7]
  n_conv_layers_list = [1, 2, 3]
  kernel_sizes = [3, 5, 7]
  strides = [1]
  paddings = [1]
  pooling_kernel_sizes = [2, 3]
  pooling_strides = [1]
  pooling_paddings = [1]
  num_layers_list = [1, 2]
  num_heads_list = [2]
  mlp_ratios = [3.0]

  for dim, box_output_size, n_conv_layers, kernel_size, stride, padding, pooling_kernel_size, pooling_stride, pooling_padding, num_layers, num_heads, mlp_ratio in product(
    dims,
    box_output_sizes,
    n_conv_layers_list,
    kernel_sizes,
    strides,
    paddings,
    pooling_kernel_sizes,
    pooling_strides,
    pooling_paddings,
    num_layers_list,
    num_heads_list,
    mlp_ratios
  ):
    model_config = {
      'dim' : dim,
      'box_output_size' : box_output_size,
      'n_conv_layers' : n_conv_layers,
      'kernel_size' : kernel_size,
      'stride' : stride,
      'padding' : padding,
      'pooling_kernel_size' : pooling_kernel_size,
      'pooling_stride' : pooling_stride,
      'pooling_padding' : pooling_padding,
      'num_layers' : num_layers,
      'num_heads' : num_heads,
      'mlp_ratio' : mlp_ratio,
    }

    print(model_config)
    train_model(model_config=model_config)