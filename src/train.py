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
  model_config = {
    'dim' : 384,
    'box_output_size' : 7,
    'n_conv_layers' : 2,
    'kernel_size' : 7,
    'stride' : 1,
    'padding' : 2,
    'pooling_kernel_size' : 3,
    'pooling_stride' : 1,
    'pooling_padding' : 1,
    'num_layers' : 14,
    'num_heads' : 6,
    'mlp_ratio' : 3.0,
  }

  train_model(model_config)