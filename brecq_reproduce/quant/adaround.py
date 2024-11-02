"""
implement of Adaptive Rounding
"""

import torch
from torch import nn
from quant.quant_layer import UniformAffineQuantizer, round_est


class AdaRoundQuantizer(nn.Module):
    """
    自适应舍入方法
    adaptive rounding quantizer, used to optimize the rounding
    policy by reconstructing the intermediate output
    """

    def __init__(self, uaq: UniformAffineQuantizer, weight_tensor: torch.Tensor, round_mode="learned_round_sigmoid"):
        super(AdaRoundQuantizer, self).__init__()
        self.n_bits = uaq.n_bits
        self.sym = uaq.sym
        self.scale = uaq.scale
        self.zero_point = uaq.zero_point
        self.n_levels = uaq.n_levels

        self.round_mode = round_mode
        self.alpha = None
        self.soft_target = False

        # sigmoid
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2/3
        self.init_alpha(x=weight_tensor.clone())

    def forward(self, x):
        if self.round_mode == "nearest":
            x_int = torch.round(x / self.scale)
        elif self.round_mode == 'nearest_ste':
            x_int = round_est(x / self.scale)
        elif self.round_mode == 'stochastic':
            x_floor = torch.floor(x / self.scale)
            rest = (x / self.scale) - x_floor
            x_int = x_floor + torch.bernoulli(rest) # 生成伯努利分布的随机样本, 随机[0, 1]
        elif self.round_mode == 'learned_hard_sigmoid':
            x_floor = torch.floor(x / self.scale)
            if self.soft_target:
                x_int = x_floor + self.get_soft_targets()
            else:
                x_int = x_floor + (self.alpha >=0).float()
        else:
            raise ValueError("")

        x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - self.zero_point) * self.scale

        return x_float_q

    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def init_alpha(self, x):
        x_floor = torch.floor(x / self.scale)
        if self.round_mode == 'learned_hard_sigmoid':
          rest = (x / self.scale) - x_floor  # rest of rounding [0, 1)
          alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1) # => sigmoid(alpha) = rest
          self.alpha = nn.Parameter(alpha)
        else:
            raise NotImplementedError
