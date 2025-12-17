import torch

from model.model import MokioMindConfig, MokioMindForCausalLM

# 创建配置和模型
cfg = MokioMindConfig(hidden_size=128, num_hidden_layers=2, num_attention_heads=4, vocab_size=1000)
model = MokioMindForCausalLM(cfg)

# 构造假的输入（batch 2，长度 5）
input_ids = torch.randint(0, cfg.vocab_size, (2, 5))

# 前向一次
out = model(input_ids=input_ids)
logits = out.logits  # [2, 5, vocab_size]
print(logits.shape)
