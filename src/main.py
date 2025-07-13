from cct_trainer import CCTTrainer

# hypothesis: it's doing well because the dimension alongside embedding is high
# kernel size - look into the convolutional tokenizer - so we retain as much as possible

def train_model(model_config):
  overrides = {
    'data': 'data/VisDrone.yaml',
    'epochs': 50,
    'batch': 8,
    'device': '0',
    'imgsz': 1920,
    'deterministic': False,
    'model_config': model_config
  }

  trainer = CCTTrainer(overrides=overrides)
  trainer.train()

if __name__ == "__main__":
  cct_configs = [
    # format: (dim, num_layers, num_heads, mlp_ratio, n_conv_layers, kernel_size, stride)
    (128, 2, 2, 2.0, 1, 1, 1),
    # (128, 4, 2, 1.0, 2, 3, 1),
    # (256, 6, 4, 2.0, 2, 3, 1),
    # (256, 7, 4, 2.0, 1, 3, 1),
    # (256, 7, 4, 2.0, 2, 7, 2)
  ]

  for config in cct_configs:
    dim, num_layers, num_heads, mlp_ratio, n_conv_layers, kernel_size, stride = config

    model_config = {
      'dim': dim,
      'box_output_size': 7,
      'n_conv_layers': n_conv_layers,
      'kernel_size': kernel_size,
      'stride': stride,
      'padding': 0,
      'pooling_kernel_size': 1,
      'pooling_stride': 1,
      'pooling_padding': 1,
      'num_layers': num_layers,
      'num_heads': num_heads,
      'mlp_ratio': mlp_ratio
    }

    print(model_config)
    train_model(model_config=model_config)