from cct_trainer import CCTTrainer

def train_model(model_config):
  overrides = {
    'data': 'data/VisDrone.yaml',
    'epochs': 1,
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
    'dim' : 384,
  }
  train_model(model_config)