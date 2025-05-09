import torch
from einops import rearrange

class CCTPredictor(torch.nn.Module):
  def __init__(self, cct_module, embed_dim, num_classes):
    super().__init__()
    self.tokenizer = cct_module.tokenizer
    self.positional_emb = cct_module.classifier.positional_emb
    self.blocks = cct_module.classifier.blocks
    self.norm = cct_module.classifier.norm
    self.attention_pool = cct_module.classifier.attention_pool
    self.cls_fc = cct_module.classifier.fc
    
    self.bbox_pred = torch.nn.Sequential(
      torch.nn.Linear(embed_dim, embed_dim // 2),
      torch.nn.ReLU(inplace=True),
      torch.nn.Linear(embed_dim // 2, num_classes * 4)
    )

  def forward(self, x):
    x = self.tokenizer(x)
    
    if self.positional_emb is not None:
      x = x + self.positional_emb
    
    for blk in self.blocks:
      x = blk(x)
      torch.cuda.empty_cache()
    
    x = self.norm(x)
    
    attn_weights = rearrange(self.attention_pool(x), 'b n 1 -> b n').softmax(dim=1)
    emb = torch.einsum('b n, b n d -> b d', attn_weights, x)
    
    cls_logits = self.cls_fc(emb)
    bbox_deltas = self.bbox_pred(emb)
    
    return cls_logits, bbox_deltas