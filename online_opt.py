import torch
from torch.optim import Optimizer
import random

import einops
from opt_einsum import contract as einsum

SMALL_VALUE = 1e-8



class OPTIMISTIC_OGD(OL_BASE):
    def __init__(self, initial_value, **kwargs):
        super().__init__(name='optimistic_ogd')
        self.iterate = initial_value
        self.lr = kwargs['lr']
        self.wd = kwargs['wd']
        self.beta = kwargs['beta']
        self.opt_iterate = initial_value.clone().detach_()

    def get_iterate(self):
        return self.opt_iterate


    @torch.no_grad()
    def update(self, grad):
        self.iterate.sub_(self.lr*(1.0-self.beta)*grad/self.beta)
        self.iterate.mul_(self.beta)
        self.opt_iterate.copy_(self.iterate*self.beta - self.lr* (1.0-self.beta)*grad)


class ADAM(OL_BASE):
    def __init__(self, initial_value, **kwargs):
        super().__init__(name='adam')
        self.iterate = initial_value
        self.lr = kwargs['lr']
        self.wd = kwargs['wd']
        self.beta2 = kwargs['beta2']
        self.beta = kwargs['beta']
        self.V = torch.zeros_like(initial_value)
        self.iterate = initial_value.clone().detach_()

    def get_iterate(self):
        return self.iterate


    @torch.no_grad()
    def update(self, grad):
        self.V.add_((1.0-self.beta2) * grad**2 / self.beta2)
        self.V.mul_(self.beta2)
        eta = self.lr * torch.rsqrt(self.V + SMALL_VALUE)
        self.iterate.sub_(eta*(1.0-self.beta)*grad/self.beta)
        self.iterate.mul_(self.beta)



class OPTIMISTIC_ADAM(OL_BASE):
    def __init__(self, initial_value, **kwargs):
        super().__init__(name='optimistic_adam')
        self.iterate = initial_value
        self.lr = kwargs['lr']
        self.wd = kwargs['wd']
        self.beta2 = kwargs['beta2']
        self.beta = kwargs['beta']
        self.V = torch.zeros_like(initial_value)
        self.opt_iterate = initial_value.clone().detach_()

    def get_iterate(self):
        return self.opt_iterate


    @torch.no_grad()
    def update(self, grad):
        self.V.add_((1.0-self.beta2) * grad**2 / self.beta2)
        self.V.mul_(self.beta2)
        eta = self.lr * torch.rsqrt(self.V + SMALL_VALUE)
        self.iterate.sub_(eta*(1.0-self.beta)*grad/self.beta)
        self.iterate.mul_(self.beta)
        self.opt_iterate.copy_(self.iterate*self.beta - eta* (1.0-self.beta)*grad)




class DynamicPF(OL_BASE):
    def __init__(self, initial_value, **kwargs):
        super().__init__(name='dynamicpf')
        self.num_lrs = kwargs['num_lrs']
        with torch.no_grad():
            self.iterates = einops.repeat(initial_value, '...  -> ... repeat', repeat=self.num_lrs).clone()
            self.alphas = torch.zeros(self.num_lrs, device=initial_value.device)
            self.G = torch.zeros(1, device=initial_value.device)
            self.V = torch.zeros(self.num_lrs, device=initial_value.device)
            self.etas = torch.tensor([kwargs['lr'] * 0.1**x for x in range(self.num_lrs)], device=initial_value.device)
            self.eps = kwargs['eps']


    def get_iterate(self):
        return torch.sum(self.iterates, dim=-1)


    @torch.no_grad()
    def update(self, grad):
        grad_norm_sq = (torch.linalg.norm(grad)**2).unsqueeze(-1)
        grad = grad.unsqueeze(-1)
        norms = torch.sqrt(einsum('... c, ... c -> c', self.iterates, self.iterates))
        theta = 2 * self.iterates *torch.log(norms/(SMALL_VALUE+self.alphas) + 1)/(SMALL_VALUE+self.etas * norms) - grad
        theta_norms = torch.sqrt(einsum('... c, ... c -> c', theta, theta))
        self.G.copy_(torch.maximum(self.G, torch.sqrt(grad_norm_sq)))
        self.V.add_(grad_norm_sq)
        self.alphas.copy_(self.eps * self.G / (SMALL_VALUE + self.V))
        self.iterates.copy_(self.alphas * theta/(theta_norms + SMALL_VALUE)*(torch.exp( self.etas * 0.5 * torch.clamp(theta_norms - 2 * self.etas * grad_norm_sq, min=0)) - 1))
    

OL_REGISTRY = {
    'ogd': OGD,
    'dynamicpf' : DynamicPF,
    'optogd': OPTIMISTIC_OGD,
    'optadam': OPTIMISTIC_ADAM,
    'adam': ADAM
}


class RandomOL(torch.optim.Optimizer):

    def __init__(self, params, ol='ogd', scale_type='random', **kwargs):
        if 'num_lrs' not in kwargs:
            kwargs['num_lrs'] = 5
        if 'eps' not in kwargs:
            kwargs['eps'] = 0.1
        super().__init__(params, kwargs)
        self.count = 0
        self.ol = ol
        self.scale_type = scale_type

        self.__setstate__(self.state)


    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['ol'] = OL_REGISTRY[self.ol](torch.zeros_like(param), **group)
                state['iterate'] = torch.clone(param).detach_()

    @torch.no_grad()
    def step(self, closure=None):
        self.count += 1
        if self.scale_type == 'random':
            scaling = random.random()
        elif self.scale_type == 'half':
            scaling = 0.5
        elif self.scale_type == 'sgd':
            scaling = 1.0

        for group in self.param_groups:
            wd = group['wd']
            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad
                grad.add_(wd * param)


                state = self.state[param]

                state['ol'].update(grad)
                delta = state['ol'].get_iterate()

                param.copy_(state['iterate'] + scaling * delta)
                state['iterate'].add_(delta)




