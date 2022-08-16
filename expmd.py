import torch
from torch.optim import Optimizer

SMALL_VALUE = 1e-8



class ExpMD(torch.optim.Optimizer):

    def __init__(self, params, lr, eps=0.01, wd=0.0, beta=0.99, momentum=False, center=False):
        super().__init__(params, {'lr': lr, 'wd': wd, 'eps': eps, 'beta': beta})
        self.center = center
        self.momentum = momentum
        self.__setstate__(self.state)


    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['second_order'] = torch.full_like(param, SMALL_VALUE)
                state['path_length'] = torch.full_like(param, SMALL_VALUE)
                if self.center:
                    state['center'] = torch.clone(param).detach()
                else:
                    state['center'] = 0.0
                # state['hint'] = torch.full_like(param, SMALL_VALUE)
                # state['V'] = torch.full_like(param, SMALL_VALUE)
                # state['b_inc'] = torch.full_like(param, 4)
                # state['B'] = torch.full_like(param, 16)
                # state['D'] = torch.full_like(param, SMALL_VALUE)
                # state['initial_param'] = param.clone().detach()


                # if self.zero_center:
                #     state['theta'] = self.get_theta(param, alpha, V, h)
                #     state['initial_param'] = torch.zeros_like(param)
                # else:
                # state['theta'] = torch.zeros_like(param)
                # state['initial_param'] = param.clone().detach()



    @torch.no_grad()
    def step(self, closure=None):

        for group in self.param_groups:
            lr = group['lr']
            wd = group['wd']
            eps = group['eps']
            beta = group['beta']
            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad

                state = self.state[param]

                offset = param - state['center']
                

                state['second_order'].add_(grad**2)# * torch.abs(offset))
                # state['second_order'].add_(grad**2 * (1-beta))# * torch.abs(offset))
                # state['second_order'].mul_(beta)

                path_length_ratio = torch.sqrt(torch.abs(offset) + lr * state['path_length'])#/(torch.abs(param) + SMALL_VALUE))

                eta = torch.minimum(path_length_ratio/torch.sqrt(state['second_order']), 0.5/(torch.abs(grad) + SMALL_VALUE))

                # compute the MD update with regularizer \psi(w) = eps/eta (w/eps + 1) log(w/eps +1) - w/eta
                # and composite term \phi(w) = eta * g^2 |w|


                x = torch.sign(offset) * torch.log(torch.abs(offset)/eps+1) - grad * eta
                x = x - torch.sign(x) * torch.minimum(grad**2 * eta**2, torch.abs(x))

                new_offset = torch.sign(x) * eps * (torch.exp(torch.abs(x)) - 1)

                state['path_length'].add_(torch.abs(new_offset - offset))


                if self.momentum:
                    state['center'] += grad**2 * (new_offset - state['center'])/state['second_order']

                param.copy_(new_offset +state['center'])

                









class ExpMDNorm(torch.optim.Optimizer):

    def __init__(self, params, lr, eps=0.01, wd=0.0, beta=0.99, center=True):
        super().__init__(params, {'lr': lr, 'wd': wd, 'eps': eps, 'beta': beta})
        self.center = center
        self.__setstate__(self.state)


    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['second_order'] = torch.zeros(1, device=param.device)
                state['path_length'] = torch.zeros(1, device=param.device)
                if self.center:
                    state['center'] = torch.clone(param).detach()
                else:
                    state['center'] = 0.0
                # state['hint'] = torch.full_like(param, SMALL_VALUE)
                # state['V'] = torch.full_like(param, SMALL_VALUE)
                # state['b_inc'] = torch.full_like(param, 4)
                # state['B'] = torch.full_like(param, 16)
                # state['D'] = torch.full_like(param, SMALL_VALUE)
                # state['initial_param'] = param.clone().detach()


                # if self.zero_center:
                #     state['theta'] = self.get_theta(param, alpha, V, h)
                #     state['initial_param'] = torch.zeros_like(param)
                # else:
                # state['theta'] = torch.zeros_like(param)
                # state['initial_param'] = param.clone().detach()



    @torch.no_grad()
    def step(self, closure=None):

        for group in self.param_groups:
            lr = group['lr']
            wd = group['wd']
            eps = group['eps']
            beta = group['beta']
            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad

                state = self.state[param]

                offset = param - state['center']
                

                state['second_order'].add_(torch.norm(grad)**2 * torch.norm(offset))
                # state['second_order'].add_(grad**2 * (1-beta))# * torch.abs(offset))
                # state['second_order'].mul_(beta)

                path_length_ratio = torch.sqrt(torch.norm(offset) + lr * state['path_length'])#/(torch.abs(param) + SMALL_VALUE))

                eta = torch.minimum(path_length_ratio/torch.sqrt(state['second_order'] + SMALL_VALUE), 0.5/(torch.norm(grad) + SMALL_VALUE))

                # compute the MD update with regularizer \psi(w) = eps/eta (w/eps + 1) log(w/eps +1) - w/eta
                # and composite term \phi(w) = eta * g^2 |w|


                x = offset / (torch.norm(offset) + SMALL_VALUE)* torch.log(torch.norm(offset)/eps+1) - grad * eta
                # x = x - x / (torch.norm(x) + SMALL_VALUE) * torch.minimum(torch.norm(grad)**2 * eta**2, torch.norm(x))

                new_offset = x / (torch.norm(x) + SMALL_VALUE) * eps * (torch.exp(torch.norm(x)) - 1)

                state['path_length'].add_(torch.norm(new_offset - offset))

                param.add_(new_offset - offset)




