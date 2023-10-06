from .functional import GradientReversalFunction
import torch
from torch import nn

class GradientReversal(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)
   

    def forward(self, x, reset_alpha):
        if not isinstance(reset_alpha, torch.Tensor):
            self.alpha = torch.tensor(reset_alpha)
        return GradientReversalFunction.apply(x, self.alpha)