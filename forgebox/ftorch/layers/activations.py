from torch import nn
import torch
import math

class GELU(nn.Module):
    """
    GELU activation instead of RELU
    <iframe src="https://www.desmos.com/calculator/fgpckn1i1m?embed" width="500px" height="500px" style="border: 1px solid #ccc" frameborder=0></iframe>
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))