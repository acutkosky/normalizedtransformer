
import torch
from torch.optim import Optimizer

SMALL_VALUE = 1e-8

class PerpOpt(torch.optim.Optimizer):

    def __init__(self, params, lr, wd=0.0, beta=0.9, alignbeta=0.9, per_layer=True):
        super().__init__(params, {'lr': lr, 'wd': wd, 'beta': beta, 'alignbeta': alignbeta})
        self.__setstate__(self.state)
        self.per_layer=per_layer


    def reset(self, epsilon=None, constraint=None, zero_center=None):
        self.__setstate__(self.state)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['momentum'] = torch.full_like(param, SMALL_VALUE)
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
            beta = group['beta']
            alignbeta = group['alignbeta']

            alignment = 0.0#torch.zeros(1)
            mom_norm_squared = 0.0#torch.zeros(1)
            if not self.per_layer:
                for param in group['params']:
                    if param.grad is None:
                        continue
                    
                    
                    grad = param.grad

                    state = self.state[param]

                    momentum = state['momentum']

                    alignment += torch.sum(grad * momentum)
                    mom_norm_squared += torch.norm(momentum)**2




            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad

                state = self.state[param]

                momentum = state['momentum']

                alignment = torch.sum(grad * momentum)
                mom_norm_squared = torch.norm(momentum)**2

                grad_perp = grad - alignment * momentum / mom_norm_squared
                grad_parallel = alignment * momentum / mom_norm_squared * (1.0 - alignbeta * (alignment < 0))


                delta = grad_perp + grad_parallel
                # delta = delta / (torch.norm(delta) + SMALL_VALUE)


                # assert torch.sum(delta * momentum) > (-1e-6 * torch.norm(momentum)*torch.norm(delta)), f"momentum not projected out: {torch.sum(delta*momentum)}, {torch.norm(momentum)}, {torch.norm(delta)}. {(-1e-6 * torch.norm(momentum)*torch.norm(delta))}!"
                # delta = delta/(torch.norm(delta)+SMALL_VALUE)

                momentum.add_(grad *(1.0-beta)/beta)
                momentum.mul_(beta)

                delta += wd * param


                param.sub_(delta * lr)




if __name__ == '__main__':

    import numpy as np
    w = torch.tensor(np.ones(1), requires_grad = True)
    
    def loss(w):
        return torch.sum(w**2)
    opt = PerpOpt([w], 0.01)
    for i in range(1000):
        l = loss(w)

        print('')
        print('iter: ', i)
        print('w: ', w)
        print('l: ', l)
        opt.zero_grad()
        l.backward()
        opt.step()
    