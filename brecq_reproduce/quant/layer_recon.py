from linecache import cache
from pickletools import optimize
from sched import scheduler

import torch
from torch.nn.utils.parametrize import cached

import linklink as link
from quant.quant_layer import QuantModule, StraightThrough, lp_norm
from quant.quant_model import QuantModel
from quant.block_recon import LinearTempDecay
from quant.adaround import AdaRoundQuantizer
from quant.data_utils import save_grad_data, save_input_output_data


def layer_reconstruction(model: QuantModel, layer: QuantModule, cali_data: torch.Tensor, batch_size: int=32, iters: int=20000, weight: float = 0.001,
                         opt_mode: str = 'mse', asym: bool = False, include_act_func: bool = True, b_range: tuple = (20, 2),
                         warmup: float = 0.0, act_quant: bool = False, lr: float = 4e-5, p: float = 2.0, multi_gpu: bool = False):
    """

    block reconstruction to optimize the output from each layer

    :param model:  quantModel
    :param layer:  quantModule that needs to be optimized
    :param cali_data: data for calibration, use adaround
    :param batch_size: mini-batch size for reconstruction
    :param iters: optimization iteration for reconstruction
    :param weight:  the weight of rounding regularization term
    :param opt_mode:  optimization mode
    :param asym:  asymmetric optimization designed in adaround, use quant input to reconstruct fp out
    :param include_act_func: optimize the output after activation function
    :param b_range: temperature range
    :param warmup: proportion of iterations that no scheduling for temperature
    :param act_quant: use activation quantization or not
    :param lr: learning rate for act delta learning
    :param p: l_p norm
    :param multi_gpu:
    :return:
    """

    model.set_quant_state(False, False)
    layer.set_quant_state(True, act_quant)
    round_mode = "learned_hard_sigmoid"

    if not include_act_func:
        org_act_func = layer.activation_function
        layer.activation_function = StraightThrough()

    if not act_quant:
        # replace weight quantizer to adaRoundQuantizer
        layer.weight_quantizer = AdaRoundQuantizer(uaq=layer.weight_quantizer, round_mode=round_mode, weight_tensor=layer.org_weight.data)
        layer.weight_quantizer.soft_target = True

        # set up optimizer
        opt_params = [layer.weight_quantizer.alpha]
        optimizer = torch.optim.Adam(opt_params)
        scheduler = None

    else:
        # use uniformAffineQuantizer to learn delta
        opt_params = [layer.activation_function.delta]
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0.)

    loss_mode = "none" if act_quant else "relaxation"
    rec_loss = opt_mode

    loss_func = LossFunction(layer, round_loss=loss_mode, weight=weight, max_count=iters, rec_loss=rec_loss, b_range=b_range, decay_start=0., warmup=warmup, p=p)

    # save data before optimization the rounding
    cache_inps, cache_outs = save_input_output_data(model, layer, cali_data, asym, act_quant, batch_size)
    if opt_mode != "mse":
        cached_grads = save_grad_data(model, layer, cali_data, act_quant, batch_size=batch_size)
    else:
        cached_grads = None
    device = 'cuda'

    for i in range(iters):
        idx = torch.randperm(cache_inps.size(0))[:batch_size]
        cur_inp = cache_inps[idx]
        cur_out = cache_outs[idx]
        cur_grad = cached_grads[idx] if opt_mode != "mse" else None

        optimizer.zero_grad()
        out_quant = layer(cur_inp)

        err = loss_func(out_quant, cur_out, cur_grad)
        err.backward()
        if multi_gpu:
            for p in opt_params:
                link.allreduce(p.grad)
        optimizer.step()
        if scheduler:
            scheduler.step()

        torch.cuda.empty_cache()

        # finish optimization, use hard rounding
        layer.weight_quantizer.soft_target = False

        # reset original activation function
        if not include_act_func:
            layer.activation_function = org_act_func


class LossFunction:
    def __init__(self,
                 layer: QuantModule,
                 round_loss: str = "relaxation",
                 weight: float = 1.,
                 rec_loss: str = "mse",
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.):
        self.layer = layer
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start, start_b=b_range[0], end_b=b_range[1])
        self.count = 0

    def __call__(self, pred, tgt, grad=None):
        """
        compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from fp model
        :param grad: gradients to computes fisher information
        :return:
        """

        self.count += 1
        if self.rec_loss == "mse":
            rec_loss = lp_norm(pred, tgt, p=self.p)
        elif self.rec_loss == 'fisher_diag':
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
        elif self.rec_loss == 'fisher_full':
            a = (pred - tgt).abs()
            grad = grad.abs()
            batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
            rec_loss = (batch_dotprod * a * grad).mean() / 100
        else:
            raise ValueError()

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            round_values = self.layer.weight_quantizer.get_soft_targets()
            round_loss += self.weight * (1 - ((round_values - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        if self.count % 500 == 0:
            print('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
                  float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return total_loss



