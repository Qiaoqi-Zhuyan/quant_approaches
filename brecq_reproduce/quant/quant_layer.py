import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


class StraightThrough(nn.Module):
    def __init__(self, channel_n: int = 1 ):
        super().__init__()

    def forward(self, x):
        return x


def round_est(x: torch.Tensor):
    """
    rounding operation
    :param x:
    :return:
    """

    return (x.round() - x).detach() + x


def lp_norm(pred, target, p=2.0, reduction='none'):
    """
    L_p norm loss function
    :param pred:
    :param target:
    :param p:
    :param reduction:
    :return:
    """

    if reduction == 'none':
        return (pred - target).abs().pow(p).sum(1).mean()
    else:
        return (pred - target).abs().pow(p).mean()


class UniformAffineQuantizer(nn.Module):
    """
    非对称量化函数, 在向前传播中量化其参数, 在后向传递中直接传递梯度, 忽略量化
    """
    def __init__(self, n_bits: int=8, symmetric=False, channel_wise=False, scale_method='max', leaf_param=False):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.scale = None
        self.zero_point = None
        self.inited = None
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method

    def forward(self, x):
        pass

    def init_quantizaton_scale(self, x, channel_wise=False):
        scale, zero_point = None, None
        if channel_wise:
            # 一个通道一个通道地决定 scale and zero point
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            scale = x_max.clone()
            zero_point = x_max.clone()
            for c in range(n_channels):
                scale[c], zero_point[c] = self.init_quantizaton_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                scale = scale.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            else:
                scale = scale.view(-1, 1)
                zero_point = zero_point.view(-1, 1)

        else:
            # 直接从tensor中的最大最小来决定scale和zero-point
            if "max" in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                x_absmax = max(abs(x_min), x_max)
                if self.sym:
                    x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax
                scale = float(x_max - x_min) / (self.n_levels - 1)
                if scale < 1e-8:
                    warnings.warn("Quantization range close to zero: [{}, {}]".format(x_min, x_max))
                    scale = 1e-8

                zero_point = round(-x_min / scale)
                scale = torch.tensor(scale).type_as(x)

            elif self.scale_method == 'mse':
                # 通过mse loss 来学习得到scale, zero-point
                x_max = x.max()
                x_min = x.min()
                best_score = 1e+10
                for i in range(80):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max, new_min)
                    # L_p norm
                    score = lp_norm(x, x_q, p=2.4, reduction="all")
                    if score < best_score:
                        best_score = score
                        scale = (new_max - new_min) / (2 ** self.n_bits - 1)
                        zero_point = (- new_min / scale).round()
            else:
                raise NotImplementedError

        return scale, zero_point


    def quantize(self, x, max, min):
        scale = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (-min / scale).round()
        x_int = torch.round(x / scale)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * scale
        return x_float_q

    def bitwidth_refactor(self, refactored_bit):
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits


class QuantModule(nn.Module):
    """
    quantized module that can perform quantized convolution or
    normal convolution, to activate quantization, please use
    set_quant_state function
    """
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant: bool=False, se_module=None):
        super(QuantModule, self).__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding, dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None

        # de-cativate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_act_quant = disable_act_quant
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

        self.se_module = se_module
        self.extra_repr = org_module.extra_repr()

    def forward(self, input: torch.Tensor):
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        # 在conv逐元素进行操作前，向禁用量化操作，进行逐元素操作后应用activation函数再quat
        if self.se_module is not None:
            out = self.se_module(out)
        out = self.activation_function(out)
        if self.disable_act_quant:
            return out
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

    def set_quant_state(self, weight_quant=False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant