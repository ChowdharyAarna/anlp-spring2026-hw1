from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            max_grad_norm: float = None,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias, max_grad_norm=max_grad_norm)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            max_grad_norm = group["max_grad_norm"]

            # TODO: Clip gradients if max_grad_norm is set
            if max_grad_norm is not None:
                params_with_grad = [p for p in group["params"] if p.grad is not None]
                if len(params_with_grad) > 0:
                    torch.nn.utils.clip_grad_norm_(params_with_grad, max_grad_norm)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]

                # TODO: Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                correct_bias = group["correct_bias"]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data) # m_0
                    state["exp_avg_sq"] = torch.zeros_like(p.data) #v_0

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                # TODO: Update first and second moments of the gradients
                old_exp_avg = exp_avg
                scaled_old_exp_avg = beta1 * old_exp_avg 
                scaled_grad_for_m = (1.0 - beta1) * grad    
                new_exp_avg = scaled_old_exp_avg + scaled_grad_for_m    

                exp_avg.copy_(new_exp_avg)

                old_exp_avg_sq = exp_avg_sq
                scaled_old_exp_avg_sq = beta2 * old_exp_avg_sq
                grad_squared = grad * grad
                scaled_grad_sq_for_v = (1.0 - beta2) * grad_squared
                new_exp_avg_sq = scaled_old_exp_avg_sq + scaled_grad_sq_for_v

                exp_avg_sq.copy_(new_exp_avg_sq)

                # TODO: Bias correction
                # Please note that we are using the "efficient version" given in Algorithm 2 
                # https://arxiv.org/pdf/1711.05101
                if correct_bias:
                    bias_correction1 = 1.0 - beta1 ** t
                    bias_correction2 = 1.0 - beta2 ** t

                    corrected_lr = lr * (bias_correction2 ** 0.5) / bias_correction1
                    step_size = corrected_lr
                else:
                    step_size = lr

                # TODO: Update parameters
                v_sqrt = exp_avg_sq.sqrt()
                denom = v_sqrt + eps 

                scaled_update = step_size * exp_avg / denom 

                p.data = p.data - scaled_update

                # TODO: Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                if weight_decay != 0.0:
                    decay_amount = lr * weight_decay
                    p.data = p.data - decay_amount * p.data

        return loss
