import torch
from torch.optim import Optimizer
import sys
import random
import wandb

SMALL_VALUE = 1e-8


def normalize(v, M=None):
    return v/(torch.linalg.norm(v) + SMALL_VALUE)

def clip_by_norm(v, max_norm):
    return v * torch.clamp(max_norm/(torch.linalg.norm(v) + SMALL_VALUE), max = 1.0)


class Nigt(torch.optim.Optimizer):

    def __init__(self, params, lr, wd=0.0, beta=0.99):
        super().__init__(params, {'lr': lr, 'wd': wd, 'beta': beta})
        self.count = 0
        self.__setstate__(self.state)


    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                # state['second_order'] = torch.full_like(param, SMALL_VALUE)
                state['momentum'] = torch.zeros_like(param)

    
    def x_to_w(self, param, state, beta, lr):
        return param + beta/(1-beta) * lr * normalize(state['momentum'])

    def w_to_x(self, param, state, beta, lr):
        return param - beta/(1-beta) * lr * normalize(state['momentum'])

    @torch.no_grad()
    def x_to_w_(self):
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            for param in group['params']:

                state = self.state[param]


                w_t = self.x_to_w(param, state, beta, lr)
                param.copy_(w_t)

    @torch.no_grad()
    def w_to_x_(self):
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            for param in group['params']:

                state = self.state[param]


                x_t = self.w_to_x(param, state, beta, lr)
                param.copy_(x_t)


    @torch.no_grad()
    def step(self, closure=None):
        self.count += 1

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad


                state = self.state[param]


                w_t = self.x_to_w(param, state, beta, lr)#param + beta/(1-beta) * lr * normalize(state['momentum'])

                state['momentum'].add_(grad * (1-beta)/beta)
                state['momentum'].mul_(beta)

                w_tplusone = w_t - lr * normalize(state['momentum'])

                x_tplusone = w_tplusone + beta/(1-beta) * (w_tplusone - w_t)

                param.copy_(x_tplusone)




class Nigt_lamb(torch.optim.Optimizer):

    def __init__(self, params, lr, wd=0.0, beta=0.99):
        super().__init__(params, {'lr': lr, 'wd': wd, 'beta': beta})
        self.count = 0
        self.__setstate__(self.state)


    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                # state['second_order'] = torch.full_like(param, SMALL_VALUE)
                state['momentum'] = torch.zeros_like(param)
                state['w_t'] = torch.clone(param).detach_()
                state['second_order'] = torch.zeros_like(param)

    
    def x_to_w(self, param, state, beta, lr):
        return param + beta/(1-beta) * state['offset']

    def w_to_x(self, param, state, beta, lr):
        return param - beta/(1-beta) *  normalize(state['offset'])

    @torch.no_grad()
    def x_to_w_(self):
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            for param in group['params']:

                state = self.state[param]


                w_t = self.x_to_w(param, state, beta, lr)
                param.copy_(w_t)

    @torch.no_grad()
    def w_to_x_(self):
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            for param in group['params']:

                state = self.state[param]


                x_t = self.w_to_x(param, state, beta, lr)
                param.copy_(x_t)


    @torch.no_grad()
    def step(self, closure=None):
        self.count += 1

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad


                state = self.state[param]

                state['second_order'].add_(grad**2 * (1-beta)/beta)
                state['second_order'].mul_(beta)

                alpha = 1.0-beta
                effective_alpha = torch.clamp(alpha/torch.sqrt(state['second_order']), max=1.0)
                effective_beta = 1.0 - effective_alpha


                w_t = self.x_to_w(param, state, effective_beta, lr)#param + beta/(1-beta) * lr * normalize(state['momentum'])

                state['momentum'].add_(grad * (1-effective_beta)/effective_beta)
                state['momentum'].mul_(effective_beta)

                state['offset'].copy_(lr * normalize(state['momentum']) * torch.linalg.norm(param))

                w_tplusone = w_t - state['offset']

                x_tplusone = w_tplusone + beta/(1-beta) * (w_tplusone - w_t)

                param.copy_(x_tplusone)





class Dynamic(torch.optim.Optimizer):

    def __init__(self, params, lr, wd=0.0, beta=0.99, implicit=False, adaptive=True):
        super().__init__(params, {'lr': lr, 'wd': wd, 'beta': beta})
        self.count = 0
        self.implicit=implicit
        self.adaptive=adaptive
        self.__setstate__(self.state)


    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                # state['second_order'] = torch.full_like(param, SMALL_VALUE)
                state['momentum'] = torch.zeros_like(param)
                state['second_order'] = torch.zeros_like(param)
                state['path_length'] = torch.zeros_like(param)

    
    def x_to_w(self, param, state, beta, lr):
        return param - state['momentum']

    def w_to_x(self, param, state, beta, lr):
        return param + state['momentum']

    @torch.no_grad()
    def x_to_w_(self):
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            for param in group['params']:

                state = self.state[param]


                w_t = self.x_to_w(param, state, beta, lr)
                param.copy_(w_t)

    @torch.no_grad()
    def w_to_x_(self):
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            for param in group['params']:

                state = self.state[param]


                x_t = self.w_to_x(param, state, beta, lr)
                param.copy_(x_t)


    @torch.no_grad()
    def step(self, closure=None):
        self.count += 1

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad


                state = self.state[param]

                eta = 1.0 - beta

                if self.implicit:
                    w = self.x_to_w(param, state, beta, lr)


    
                if self.adaptive:

                    state['second_order'].add_(grad**2 + torch.abs(grad*lr))
                    state['path_length'].add(torch.abs(state['momentum']))
                    ada_scale = torch.sqrt((state['path_length']*lr + lr**2 +SMALL_VALUE)/(state['second_order'] + SMALL_VALUE))
                else:
                    ada_scale = 1.0
                
                state['momentum'].sub_(eta * grad * ada_scale)
                state['momentum'].copy_(clip_by_norm(state['momentum'], lr))


                
                if self.implicit:
                    w.add_(state['momentum'])
                    param.copy_(w+state['momentum'])
                else:
                    param.add_(state['momentum'])



class Dynamic_reg(torch.optim.Optimizer):

    def __init__(self, params, lr, wd=0.0, beta=0.99, implicit=False, adaptive=True):
        super().__init__(params, {'lr': lr, 'wd': wd, 'beta': beta})
        self.count = 0
        self.implicit=implicit
        self.adaptive=adaptive
        self.__setstate__(self.state)


    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                # state['second_order'] = torch.full_like(param, SMALL_VALUE)
                state['momentum'] = torch.zeros_like(param)
                state['second_order'] = torch.zeros_like(param)
                state['path_length'] = torch.zeros_like(param)

    
    def x_to_w(self, param, state, beta, lr):
        return param - state['momentum']

    def w_to_x(self, param, state, beta, lr):
        return param + state['momentum']

    @torch.no_grad()
    def x_to_w_(self):
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            for param in group['params']:

                state = self.state[param]


                w_t = self.x_to_w(param, state, beta, lr)
                param.copy_(w_t)

    @torch.no_grad()
    def w_to_x_(self):
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            for param in group['params']:

                state = self.state[param]


                x_t = self.w_to_x(param, state, beta, lr)
                param.copy_(x_t)


    @torch.no_grad()
    def step(self, closure=None):
        self.count += 1

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad


                state = self.state[param]

                eta = 1.0 - beta

                if self.implicit:
                    w = self.x_to_w(param, state, beta, lr)

    
                if self.adaptive:

                    state['second_order'].add_(grad**2 + torch.abs(grad*lr))
                    state['path_length'].add(torch.abs(state['momentum']))
                    ada_scale = torch.sqrt((state['path_length']*lr + lr**2 +SMALL_VALUE)/(state['second_order'] + SMALL_VALUE))
                    # ada_scale = torch.rsqrt(state['second_order'] + SMALL_VALUE)
                else:
                    ada_scale = 1.0
                
                state['momentum'].sub_(eta * grad * ada_scale)
                state['momentum'].div_(1.0 + eta * ada_scale/lr)


                
                if self.implicit:
                    w.add_(state['momentum'])
                    param.copy_(w+state['momentum'])
                else:
                    param.add_(state['momentum'])






class Dynamic_reg_reset(torch.optim.Optimizer):

    def __init__(self, params, lr, wd=0.0, beta=0.99, implicit=False, adaptive=True):
        super().__init__(params, {'lr': lr, 'wd': wd, 'beta': beta})
        self.count = 0
        self.implicit=implicit
        self.adaptive=adaptive
        self.__setstate__(self.state)


    def do_reset(self, threshold, state, param):
        if (torch.linalg.norm(param - state['center']) > threshold):
            # state['momentum'] = torch.zeros_like(param)
            state['second_order'] = torch.zeros_like(param)
            state['path_length'] = torch.zeros_like(param)
            state['center'] = torch.clone(param).detach_()


    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                # state['second_order'] = torch.full_like(param, SMALL_VALUE)
                state['momentum'] = torch.zeros_like(param)
                state['second_order'] = torch.zeros_like(param)
                state['path_length'] = torch.zeros_like(param)
                state['center'] = torch.clone(param).detach_()

    
    def x_to_w(self, param, state, beta, lr):
        return param - state['momentum']

    def w_to_x(self, param, state, beta, lr):
        return param + state['momentum']

    @torch.no_grad()
    def x_to_w_(self):
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            for param in group['params']:

                state = self.state[param]


                w_t = self.x_to_w(param, state, beta, lr)
                param.copy_(w_t)

    @torch.no_grad()
    def w_to_x_(self):
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            for param in group['params']:

                state = self.state[param]


                x_t = self.w_to_x(param, state, beta, lr)
                param.copy_(x_t)


    @torch.no_grad()
    def step(self, closure=None):
        self.count += 1

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad


                state = self.state[param]

                # self.do_reset(10*lr, state, param)

                eta = 1.0# - beta

                if self.implicit:
                    w = self.x_to_w(param, state, beta, lr)

                beta2 = beta#1.0-lr*(1.0-beta)
                if self.adaptive:

                    state['second_order'].add_((1-beta2) * (grad**2 + torch.abs(grad*lr))/beta2)
                    state['second_order'].mul_(beta2)
                    state['path_length'].add((1-beta2) *torch.abs(state['momentum'])/beta2)
                    state['path_length'].mul(beta2)
                    # ada_scale = torch.sqrt((state['path_length']*lr + lr**2 +SMALL_VALUE)/(state['second_order'] + SMALL_VALUE))
                    ada_scale = lr*torch.rsqrt(state['second_order'] + SMALL_VALUE)
                else:
                    ada_scale = 1.0
                
                state['momentum'].sub_(eta * grad * ada_scale)
                state['momentum'].div_(1.0 + eta * ada_scale/lr)


                
                if self.implicit:
                    w.add_(state['momentum'])
                    param.copy_(w+state['momentum'])
                else:
                    param.add_(state['momentum'])

