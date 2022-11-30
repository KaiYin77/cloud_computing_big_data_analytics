import torch
from torch import nn
from torch.nn import functional as F

class TimestepEmbedding(nn.Module):
    def __init__(self, time_steps, embed_dim) -> None:
        super().__init__()

    def forward(self, x: torch.IntTensor | torch.LongTensor) -> torch.Tensor:
        return self.embedding(x)
