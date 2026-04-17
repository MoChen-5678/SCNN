conda activate torch_env 环境指令
torchrun --standalone --nproc_per_node=2 Train.py 训练指令
最佳参数
{
  "final_config": {
    "learning_rate": 0.0002,
    "batch_size": 64,
    "hidden_dim": 128,
    "gru_hidden": 512,
    "modes": 16,
    "weight_decay": 0.0,无意义
    "lambda_phy_max": 0.1,无意义
    "phy_warmup": 80,
    "seq_len": 8,无意义
    "clip_grad": 2.0
  },