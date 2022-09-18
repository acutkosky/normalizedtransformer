import torch
from torch.optim import Optimizer
import random
import numpy as np
from optional_module import NoOpModule
import einops
from opt_einsum import contract as einsum

SMALL_VALUE = 1e-8

class OL_BASE(object):
    def __init__(self, name):
        self.name = name


    def get_iterate(self):
        raise NotImplementedError
    
    def update(self, grad, *args, **kwargs):
        raise NotImplementedError
    def get_logging_info(self, param, grad, local_states, logging_info):
        return None
    
    def global_update(self, param, grad, local_states):
        return None



class OGD(OL_BASE):
    def __init__(self, initial_value, **kwargs):
        super().__init__(name='ogd')
        self.iterate = initial_value
        self.lr = kwargs['lr']
        self.wd = kwargs['wd']

        self.beta = kwargs['beta']

    def get_iterate(self):
        return self.iterate


    @torch.no_grad()
    def update(self, grad, *args, **kwargs):
        self.iterate.sub_(self.lr*(1.0-self.beta)*grad/self.beta)
        self.iterate.mul_(self.beta)

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
    def update(self, grad, *args, **kwargs):
        self.iterate.sub_(self.lr*(1.0-self.beta)*grad/self.beta)
        self.iterate.mul_(self.beta)
        self.opt_iterate.copy_(self.iterate*self.beta - self.lr* (1.0-self.beta)*grad)


class SCALE_ADAM(OL_BASE):
    def __init__(self, initial_value, **kwargs):
        super().__init__(name='adam')
        self.iterate = initial_value
        self.lr = kwargs['lr']
        self.wd = kwargs['wd']
        self.beta2 = kwargs['beta2']
        self.beta = kwargs['beta']
        self.V = torch.zeros_like(initial_value)
        self.m = torch.zeros_like(initial_value)
        self.iterate = initial_value.clone().detach_()
        self.scaling = torch.ones(1, device=initial_value.device) * self.lr
        self.scale_V = torch.zeros(1, device=initial_value.device)
        self.scale_eta = self.lr
        self.count = 0

    def get_iterate(self):
        return self.iterate


    @torch.no_grad()
    def update(self, grad, *args, **kwargs):

        # metag = torch.sum(grad * self.iterate)/self.scaling
        metag = -torch.sum(grad * self.m * torch.rsqrt(self.V + SMALL_VALUE))
        # assert torch.abs(metag_p- metag)<0.001, f"uh oh: {metag_p}, {metag}"


        self.count += 1

        self.V.add_((1.0-self.beta2) * grad**2 / self.beta2)
        self.V.mul_(self.beta2)
        self.m.add_((1.0-self.beta) * grad / self.beta)
        self.m.mul_(self.beta)

        V = self.V#/(1.0-self.beta2**self.count)
        m = self.m#/(1.0-self.beta**self.count)

        # self.V.add_((1.0-self.beta2) * grad**2 / self.beta2)
        # self.V.mul_(self.beta2)
        # self.m.add_((1.0-self.beta) * grad / self.beta)
        # self.m.mul_(self.beta)
        # self.iterate.copy_(-self.lr * self.m *torch.rsqrt(self.V + SMALL_VALUE))
        
        self.scale_V.mul_(self.beta2)
        self.scale_V.add_(metag**2)
        

        self.scaling.mul_(torch.exp(-metag * torch.rsqrt(self.scale_V+SMALL_VALUE) - metag**2 /(self.scale_V + SMALL_VALUE)))
        # eta = self.lr * torch.rsqrt(self.V + SMALL_VALUE)
        self.iterate.copy_(-self.scaling* m * torch.rsqrt(V + SMALL_VALUE))
#############



class SCALE_ADAM_CLIP(OL_BASE):
    def __init__(self, initial_value, **kwargs):
        super().__init__(name='adam')
        self.iterate = initial_value
        self.lr = kwargs['lr']
        self.wd = kwargs['wd']
        self.beta2 = kwargs['beta2']
        self.beta = kwargs['beta']
        self.V = torch.zeros_like(initial_value)
        self.m = torch.zeros_like(initial_value)
        self.iterate = initial_value.clone().detach_()
        self.scaling = torch.ones(1, device=initial_value.device) * self.lr
        self.scale_V = torch.zeros(1, device=initial_value.device)
        self.scale_eta = self.lr
        self.count = 0

    def get_iterate(self):
        return self.iterate


    @torch.no_grad()
    def update(self, grad, *args, **kwargs):

        # metag = torch.sum(grad * self.iterate)/self.scaling
        metag = -torch.sum(grad * self.m * torch.rsqrt(self.V + SMALL_VALUE))
        # assert torch.abs(metag_p- metag)<0.001, f"uh oh: {metag_p}, {metag}"


        self.count += 1

        self.V.add_((1.0-self.beta2) * grad**2 / self.beta2)
        self.V.mul_(self.beta2)
        self.m.add_((1.0-self.beta) * grad / self.beta)
        self.m.mul_(self.beta)

        V = self.V#/(1.0-self.beta2**self.count)
        m = self.m#/(1.0-self.beta**self.count)

        # self.V.add_((1.0-self.beta2) * grad**2 / self.beta2)
        # self.V.mul_(self.beta2)
        # self.m.add_((1.0-self.beta) * grad / self.beta)
        # self.m.mul_(self.beta)
        # self.iterate.copy_(-self.lr * self.m *torch.rsqrt(self.V + SMALL_VALUE))
        
        self.scale_V.mul_(self.beta2)
        self.scale_V.add_(metag**2)
        

        self.scaling.mul_(torch.exp(-metag * torch.rsqrt(self.scale_V+SMALL_VALUE) - metag**2 /(self.scale_V + SMALL_VALUE)))
        self.scaling.clamp_(1e-10, 0.1)
        # eta = self.lr * torch.rsqrt(self.V + SMALL_VALUE)
        self.iterate.copy_(-self.scaling* m * torch.rsqrt(V + SMALL_VALUE))
#############


class ORIG_ADAM(OL_BASE):
    def __init__(self, initial_value, **kwargs):
        super().__init__(name='adam')
        self.iterate = initial_value
        self.lr = kwargs['lr']
        self.wd = kwargs['wd']
        self.beta2 = kwargs['beta2']
        self.beta = kwargs['beta']
        self.V = torch.zeros_like(initial_value)
        self.m = torch.zeros_like(initial_value)
        self.iterate = initial_value.clone().detach_()

    def get_iterate(self):
        return self.iterate


    @torch.no_grad()
    def update(self, grad, *args, **kwargs):
        self.V.add_((1.0-self.beta2) * grad**2 / self.beta2)
        self.V.mul_(self.beta2)
        self.m.add_((1.0-self.beta) * grad / self.beta)
        self.m.mul_(self.beta)
        eta = self.lr * torch.rsqrt(self.V + SMALL_VALUE)
        self.iterate.copy_(-self.lr * self.m *torch.rsqrt(self.V + SMALL_VALUE))


class SCALE_ADAM_CLIP_GLOBAL(OL_BASE):
    def __init__(self, initial_value, **kwargs):
        super().__init__(name='adam')
        self.iterate = initial_value
        self.lr = kwargs['lr']
        self.wd = kwargs['wd']
        self.beta2 = kwargs['beta2']
        self.beta = kwargs['beta']
        self.beta3 = kwargs['beta3']
        self.V = torch.zeros_like(initial_value)
        self.m = torch.zeros_like(initial_value)
        self.iterate = initial_value.clone().detach_()
        self.scaling = torch.ones(1, device=initial_value.device) * self.lr
        self.scale_V = torch.zeros(1, device=initial_value.device)
        self.scale_eta = self.lr
        self.count = 0

    def get_iterate(self):
        return self.iterate


    @torch.no_grad()
    def update(self, grad, *args, **kwargs):

        # metag = torch.sum(grad * self.iterate)/self.scaling
        metag = -torch.sum(grad * self.m * torch.rsqrt(self.V + SMALL_VALUE))
        return metag

    def get_logging_info(self, param, grad, local_states, logging_info):
        metag = 0
        for k, v in local_states.items():
            metag += v
        return {'optimizer/learned_lr': self.scaling, 'optimizer/meta_gradient': metag}


    @torch.no_grad()
    def global_update(self, param, grad, local_states):
        metag = 0
        for k, v in local_states.items():
            metag += v
        # assert torch.abs(metag_p- metag)<0.001, f"uh oh: {metag_p}, {metag}"


        self.count += 1

        self.V.add_((1.0-self.beta2) * grad**2 / self.beta2)
        self.V.mul_(self.beta2)
        self.m.add_((1.0-self.beta) * grad / self.beta)
        self.m.mul_(self.beta)

        V = self.V#/(1.0-self.beta2**self.count)
        m = self.m#/(1.0-self.beta**self.count)

        # self.V.add_((1.0-self.beta2) * grad**2 / self.beta2)
        # self.V.mul_(self.beta2)
        # self.m.add_((1.0-self.beta) * grad / self.beta)
        # self.m.mul_(self.beta)
        # self.iterate.copy_(-self.lr * self.m *torch.rsqrt(self.V + SMALL_VALUE))
        
        self.scale_V.mul_(self.beta3)
        self.scale_V.add_(metag**2)
        

        self.scaling.mul_(torch.exp(-metag * torch.rsqrt(self.scale_V+SMALL_VALUE) - metag**2 /(self.scale_V + SMALL_VALUE)))
        self.scaling.clamp_(1e-10, 0.1)
        # logger.log({
        #     'optimizer/iterate': self.scaling,
        # },
        # commit=False)
        # eta = self.lr * torch.rsqrt(self.V + SMALL_VALUE)
        self.iterate.copy_(-self.scaling* m * torch.rsqrt(V + SMALL_VALUE))
#############






class SCALE_ADAM_MANYLR(OL_BASE):
    def __init__(self, initial_value, **kwargs):
        super().__init__(name='adam')
        self.iterate = initial_value
        self.lr = kwargs['lr']
        self.wd = kwargs['wd']
        self.beta2 = kwargs['beta2']
        self.beta = kwargs['beta']
        self.V = torch.zeros_like(initial_value)
        self.m = torch.zeros_like(initial_value)
        self.iterate = initial_value.clone().detach_()
        self.num_lrs = 8

        self.scaling_lrs = [10**-k for k in range(self.num_lrs)]
        self.scaling = [torch.ones(1, device=initial_value.device) * self.lr/self.num_lrs * s for s in self.scaling_lrs]
        self.lower_bound = self.lr/self.num_lrs * 0.1

        self.scale_V = torch.zeros(1, device=initial_value.device)
        self.scale_eta = self.lr
        self.count = 0

    def get_iterate(self):
        return self.iterate


    @torch.no_grad()
    def update(self, grad, *args, **kwargs):

        # metag = torch.sum(grad * self.iterate)/self.scaling
        metag = -torch.sum(grad * self.m * torch.rsqrt(self.V + SMALL_VALUE))
        # assert torch.abs(metag_p- metag)<0.001, f"uh oh: {metag_p}, {metag}"


        self.count += 1

        self.V.add_((1.0-self.beta2) * grad**2 / self.beta2)
        self.V.mul_(self.beta2)
        self.m.add_((1.0-self.beta) * grad / self.beta)
        self.m.mul_(self.beta)

        V = self.V#/(1.0-self.beta2**self.count)
        m = self.m#/(1.0-self.beta**self.count)

        # self.V.add_((1.0-self.beta2) * grad**2 / self.beta2)
        # self.V.mul_(self.beta2)
        # self.m.add_((1.0-self.beta) * grad / self.beta)
        # self.m.mul_(self.beta)
        # self.iterate.copy_(-self.lr * self.m *torch.rsqrt(self.V + SMALL_VALUE))
        s_lr_base = torch.rsqrt(self.scale_V+SMALL_VALUE)
        for s_lr, s in zip(self.scaling_lrs, self.scaling):
            s_eta = s_lr*s_lr_base
            s.mul_(torch.exp(-metag * s_eta - metag**2 *s_eta**2))
            s.clamp_(self.lower_bound, s_lr* self.lr/self.num_lrs * 10)

        self.scale_V.mul_(self.beta2)
        self.scale_V.add_(metag**2)
        

        self.scaling.mul_(torch.exp(-metag * torch.rsqrt(self.scale_V+SMALL_VALUE) - metag**2 /(self.scale_V + SMALL_VALUE)))
        # eta = self.lr * torch.rsqrt(self.V + SMALL_VALUE)
        self.iterate.copy_(-self.scaling* m * torch.rsqrt(V + SMALL_VALUE))
#############



class SCALE_ADAM_DIAG(OL_BASE):
    def __init__(self, initial_value, **kwargs):
        super().__init__(name='adam')
        self.iterate = initial_value
        self.lr = kwargs['lr']
        self.wd = kwargs['wd']
        self.beta2 = kwargs['beta2']
        self.beta = kwargs['beta']
        self.V = torch.zeros_like(initial_value)
        self.m = torch.zeros_like(initial_value)
        self.iterate = initial_value.clone().detach_()
        self.scaling = torch.full_like(initial_value, fill_value=self.lr)
        self.scale_V = torch.zeros_like(initial_value)
        self.scale_eta = self.lr
        self.count = 0

    def get_iterate(self):
        return self.iterate


    @torch.no_grad()
    def update(self, grad, *args, **kwargs):

        # metag = torch.sum(grad * self.iterate)/self.scaling
        metag = -grad * self.m * torch.rsqrt(self.V + SMALL_VALUE)
        # assert torch.abs(metag_p- metag)<0.001, f"uh oh: {metag_p}, {metag}"


        self.count += 1

        self.V.add_((1.0-self.beta2) * grad**2 / self.beta2)
        self.V.mul_(self.beta2)
        self.m.add_((1.0-self.beta) * grad / self.beta)
        self.m.mul_(self.beta)

        V = self.V#/(1.0-self.beta2**self.count)
        m = self.m#/(1.0-self.beta**self.count)

        # self.V.add_((1.0-self.beta2) * grad**2 / self.beta2)
        # self.V.mul_(self.beta2)
        # self.m.add_((1.0-self.beta) * grad / self.beta)
        # self.m.mul_(self.beta)
        # self.iterate.copy_(-self.lr * self.m *torch.rsqrt(self.V + SMALL_VALUE))
        
        self.scale_V.mul_(self.beta2)
        self.scale_V.add_(metag**2)
        

        self.scaling.mul_(torch.exp(-metag * torch.rsqrt(self.scale_V+SMALL_VALUE) - metag**2 /(self.scale_V + SMALL_VALUE)))
        # eta = self.lr * torch.rsqrt(self.V + SMALL_VALUE)
        self.iterate.copy_(-self.scaling* m * torch.rsqrt(V + SMALL_VALUE))
#############


class OPTIMISTIC_CONSTRAINED_ADAGRAD(OL_BASE):
    def __init__(self, initial_value, **kwargs):
        super().__init__(name='optimistic_ogd')
        self.iterate = initial_value
        self.lr = kwargs['lr']
        self.wd = kwargs['wd']
        self.beta = kwargs['beta2']
        self.V = torch.zeros_like(initial_value)
        self.h = torch.zeros_like(initial_value)
        self.opt_iterate = initial_value.clone().detach_()

    def get_iterate(self):
        return self.opt_iterate


    @torch.no_grad()
    def update(self, grad, *args, **kwargs):
        self.V.add_((self.h - grad)**2*(1.0-self.beta)/self.beta)
        self.h.copy_(grad)
        self.V.mul_(self.beta)
        self.iterate.sub_((torch.abs(self.iterate) + 0.01*self.lr)*grad * torch.rsqrt(self.V + SMALL_VALUE))
        self.iterate.copy_(torch.clamp(self.iterate, -self.lr, self.lr))
        self.opt_iterate.copy_(torch.clamp(self.iterate - (torch.abs(self.iterate) + 0.01*self.lr) * self.h * torch.rsqrt(self.V + SMALL_VALUE), -self.lr, self.lr))


def normalize(v):
    return v/(torch.linalg.norm(v)+SMALL_VALUE)
def clip_by_norm(v, threshold):
    return v*torch.clamp(threshold/(torch.linalg.norm(v)+SMALL_VALUE),0, 1)

class CONSTRAINED_L2_ADAGRAD(OL_BASE):
    def __init__(self, initial_value, **kwargs):
        super().__init__(name='optimistic_ogd')
        self.iterate = initial_value
        self.lr = kwargs['lr']
        self.wd = kwargs['wd']
        self.beta = kwargs['beta']
        self.V = torch.zeros_like(initial_value)
        # self.V = torch.zeros(1, device=initial_value.device)

    def get_iterate(self):
        return self.iterate


    @torch.no_grad()
    def update(self, grad, *args, **kwargs):
        if torch.linalg.norm(self.iterate) > self.lr and torch.sum(grad * self.iterate)<0:
            grad.sub_(torch.sum(grad * self.iterate) * normalize(self.iterate))
        self.V.add_(grad**2*(1.0-self.beta)/self.beta)
        self.V.mul_(self.beta)
        self.iterate.sub_(self.lr*grad * torch.rsqrt(self.V + SMALL_VALUE))
        self.iterate.copy_(clip_by_norm(self.iterate, self.lr))


class CONSTRAINED_ADAGRAD(OL_BASE):
    def __init__(self, initial_value, **kwargs):
        super().__init__(name='optimistic_ogd')
        self.iterate = initial_value
        self.lr = kwargs['lr']
        self.wd = kwargs['wd']
        self.beta = kwargs['beta2']
        self.V = torch.zeros_like(initial_value)
        self.sum_G = torch.zeros_like(initial_value)

    def get_iterate(self):
        return self.iterate


    @torch.no_grad()
    def update(self, grad, *args, **kwargs):
        self.V.add_( grad**2*(1.0-self.beta)/self.beta)
        self.V.mul_(self.beta)
        self.sum_G.add_( grad*(1.0-self.beta)/self.beta)
        self.sum_G.mul_(self.beta)
        # self.iterate.sub_(self.lr*grad * torch.rsqrt(self.V + SMALL_VALUE))
        # self.iterate.copy_(torch.clamp(self.iterate, -self.lr, self.lr))
        self.iterate.copy_(torch.clamp(-self.lr * self.sum_G*torch.rsqrt(self.V + SMALL_VALUE), -self.lr, self.lr))


class REWARD_RESET(OL_BASE):
    def __init__(self, initial_value, **kwargs):
        super().__init__(name='optimistic_ogd')
        self.iterate = initial_value
        self.lr = kwargs['lr']
        self.wd = kwargs['wd']
        self.beta = kwargs['beta2']
        self.V = torch.zeros_like(initial_value)
        self.reward = torch.zeros(1, device=initial_value.device)
        self.sum_G = torch.zeros_like(initial_value)
        self.count = 0

    def get_iterate(self):
        return self.iterate

    def reset(self):
        self.iterate = torch.zeros_like(self.iterate)
        self.V = torch.zeros_like(self.V)
        self.reward = torch.zeros_like(self.reward)


    @torch.no_grad()
    def update(self, grad, *args, **kwargs):
        self.count += 1
        self.reward.add_(torch.sum(self.iterate * grad))
        if self.reward<-4*np.sqrt(2)*self.lr*torch.sum(torch.sqrt(self.V + grad**2)):
            self.reset()
            print("reset!")

        self.sum_G.add_(grad)
        comparator = self.lr * torch.sum(torch.abs(self.sum_G))
        bound = 2*np.sqrt(2.0) * self.lr*torch.sum(torch.sqrt(self.V + grad**2))
        assert self.reward + comparator <= bound, f"violating regret! cound: {self.count}, reward: {self.reward.item()}, compare: {comparator}, bound: {bound}, diff: {(bound - (self.reward + comparator)).item()}"
        if (self.count % 100 == 0):
            print(f"reward: {self.reward.item()}, comparator: {comparator}, bound: {bound}")
        self.V.add_( grad**2)#*(1.0-self.beta)/self.beta)
        # self.V.mul_(self.beta)
        self.iterate.sub_(2*self.lr*grad * torch.rsqrt(2*self.V + SMALL_VALUE))
        self.iterate.copy_(torch.clamp(self.iterate, -self.lr, self.lr))



class REWARD_RESET_CB(OL_BASE):
    def __init__(self, initial_value, **kwargs):
        super().__init__(name='optimistic_ogd')
        self.iterate = initial_value
        self.lr = kwargs['lr']
        self.wd = kwargs['wd']
        self.beta = kwargs['beta2']
        self.V = torch.zeros_like(initial_value)
        self.reward = torch.zeros(1, device=initial_value.device)
        self.sum_G = torch.zeros_like(initial_value)
        self.max_G = torch.zeros_like(initial_value)
        self.cb_reward = torch.zeros_like(initial_value)
        self.count = 0

    def get_iterate(self):
        return self.iterate

    def reset(self):
        self.iterate = torch.zeros_like(self.iterate)
        self.V = torch.zeros_like(self.V)
        self.reward = torch.zeros_like(self.reward)
        self.cb_reward = torch.zeros_like(self.cb_reward)
        self.sum_G = torch.zeros_like(self.sum_G)


    @torch.no_grad()
    def update(self, grad,  *args, **kwargs):
        self.count += 1
        self.reward.add_(torch.sum(self.iterate * grad))
        # size = torch.
        if self.reward<- 2*self.lr* torch.sum(self.max_G):
            self.reset()
            print("reset!")


        # grad = grad * (torch.logical_or(torch.abs(self.iterate) < self.lr, self.iterate * grad >0))
        g_clip = torch.clamp(grad, -self.max_G, self.max_G)
        # g_clip = grad
        self.max_G = torch.maximum(self.max_G, torch.abs(grad))
        self.cb_reward.add_(self.iterate * g_clip)
        self.V.add_(g_clip**2)
        self.sum_G.add_(g_clip)
        self.iterate.copy_(-(self.cb_reward + self.max_G * self.lr) *torch.clamp(self.sum_G / (2*self.V + 2*self.max_G**2 +SMALL_VALUE), -0.5/self.max_G, 0.5 / self.max_G))
        assert torch.sum(self.cb_reward) <= self.lr * torch.sum(self.max_G), f"uh oh: cb: {torch.sum(self.cb_reward)}, max_g: {self.lr*torch.sum(self.max_G)}, dim: {self.max_G.view(-1).size()}"

        # self.sum_G.add_(grad)
        # comparator = self.lr * torch.sum(torch.abs(self.sum_G))
        # bound = 2*np.sqrt(2.0) * self.lr*torch.sum(torch.sqrt(self.V + grad**2))
        # assert self.reward + comparator <= bound, f"violating regret! cound: {self.count}, reward: {self.reward.item()}, compare: {comparator}, bound: {bound}, diff: {(bound - (self.reward + comparator)).item()}"
        # if (self.count % 100 == 0):
        #     print(f"reward: {self.reward.item()}, comparator: {comparator}, bound: {bound}")
        # self.V.add_( grad**2)#*(1.0-self.beta)/self.beta)
        # # self.V.mul_(self.beta)
        # self.iterate.sub_(2*self.lr*grad * torch.rsqrt(2*self.V + SMALL_VALUE))
        # self.iterate.copy_(torch.clamp(self.iterate, -self.lr, self.lr))


class REWARD_RESET_PF(OL_BASE):
    def __init__(self, initial_value, **kwargs):
        super().__init__(name='optimistic_ogd')
        self.iterate = initial_value
        self.lr = kwargs['lr']
        self.wd = kwargs['wd']
        self.beta = kwargs['beta2']
        self.V = torch.zeros_like(initial_value)
        self.reward = torch.zeros(1, device=initial_value.device)
        self.sum_G = torch.zeros_like(initial_value)
        self.max_G = torch.zeros_like(initial_value)
        self.cb_reward = torch.zeros_like(initial_value)
        self.count = 0

    def get_iterate(self):
        return self.iterate

    def reset(self):
        self.iterate = torch.zeros_like(self.iterate)
        self.V = torch.zeros_like(self.V)
        self.reward = torch.zeros_like(self.reward)
        self.cb_reward = torch.zeros_like(self.cb_reward)
        self.sum_G = torch.zeros_like(self.sum_G)


    @torch.no_grad()
    def update(self, grad, *args, **kwargs):
        self.count += 1
        self.max_G = torch.maximum(self.max_G, torch.abs(grad))
        self.V.add_(grad**2*(1.0-self.beta)/self.beta)
        self.V.mul_(self.beta)
        self.sum_G.add_(grad*(1.0-self.beta)/self.beta)
        self.sum_G.mul_(self.beta)
        self.iterate.copy_(torch.clamp(-self.lr*torch.sign(self.sum_G) * (torch.exp(torch.abs(self.sum_G) * torch.rsqrt(self.V + SMALL_VALUE)) - 1), -self.lr, self.lr))



class SCALE_PF(OL_BASE):
    def __init__(self, initial_value, **kwargs):
        super().__init__(name='optimistic_ogd')
        self.iterate = initial_value
        self.lr = kwargs['lr']
        self.wd = kwargs['wd']
        self.beta = kwargs['beta']
        self.beta2 = kwargs['beta2']
        self.V = torch.zeros_like(initial_value)
        self.reward = torch.zeros(1, device=initial_value.device)
        self.sum_G = torch.zeros_like(initial_value)
        self.max_G = torch.zeros_like(initial_value)
        self.cb_reward = torch.zeros_like(initial_value)
        self.count = 0
        self.m = torch.zeros_like(initial_value)
        self.scale = torch.full_like(initial_value, fill_value=self.lr)
        self.scale_n = torch.full_like(initial_value, fill_value=self.lr)

    def get_iterate(self):
        return self.iterate

    def reset(self):
        self.iterate = torch.zeros_like(self.iterate)


        self.V = torch.zeros_like(self.V)
        self.reward = torch.zeros_like(self.reward)
        self.cb_reward = torch.zeros_like(self.cb_reward)
        self.sum_G = torch.zeros_like(self.sum_G)


    @torch.no_grad()
    def update(self, grad, *args, **kwargs):
        self.count += 1

        metag = -grad * self.m

        # self.m.add_((1.0-self.beta) * grad / self.beta)
        # self.m.mul_(self.beta)

        # D(x,y) = 1/eta (xlog(x/a) - x/eta) - 1/eta(ylogy/a -y/eta) - (x-y)/etalog(y/a) = xlog(x/y)/eta -x/eta+y/eta
        # log(x/a) - log(y/a) + eta g = 0
        # log(x/a) = log(y/a) - eta g
        # x = y * exp(-eta g)

        self.max_G = torch.maximum(self.max_G, torch.abs(metag))
        self.V.add_(metag**2*(1.0-self.beta2)/self.beta2)
        self.V.mul_(self.beta2)
        V = self.V
        V = self.V/(1.0-self.beta2)#**self.count
        self.sum_G.add_(metag*(1.0-self.beta)/self.beta)
        self.sum_G.mul_(self.beta)

        self.scale.mul_(torch.exp(-metag*torch.rsqrt(V + SMALL_VALUE) - metag**2/(V + SMALL_VALUE)))
        self.scale_n.mul_(torch.exp(metag*torch.rsqrt(V + SMALL_VALUE) - metag**2/(V + SMALL_VALUE)))
        self.iterate.copy_(self.scale * self.m)
        # self.iterate.copy_(torch.clamp(-self.lr*torch.sign(self.sum_G) * (torch.exp(torch.abs(self.sum_G) * torch.rsqrt(self.V + SMALL_VALUE)) - 1), -self.lr, self.lr))
        

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
    def update(self, grad, *args, **kwargs):
        self.V.add_((1.0-self.beta2) * grad**2 / self.beta2)
        self.V.mul_(self.beta2)
        eta = self.lr * torch.rsqrt(self.V + SMALL_VALUE)
        self.iterate.sub_(eta*(1.0-self.beta)*grad/self.beta)
        self.iterate.mul_(self.beta)



class ADAM_DA(OL_BASE):
    def __init__(self, initial_value, **kwargs):
        super().__init__(name='adam')
        self.iterate = initial_value
        self.lr = kwargs['lr']
        self.wd = kwargs['wd']
        self.beta2 = kwargs['beta2']
        self.beta = kwargs['beta']
        self.V = torch.zeros_like(initial_value)
        self.sum_G = torch.zeros_like(initial_value)
        self.iterate = initial_value.clone().detach_()

    def get_iterate(self):
        return self.iterate


    @torch.no_grad()
    def update(self, grad, *args, **kwargs):
        self.V.add_((1.0-self.beta2) * grad**2 / self.beta2)
        self.V.mul_(self.beta2)
        self.sum_G.add_((1.0-self.beta2) * grad / self.beta2)
        self.sum_G.mul_(self.beta2)
        eta = self.lr * torch.rsqrt(self.V + SMALL_VALUE)
        self.iterate.copy_(-self.sum_G * eta)
        # self.iterate.sub_(eta*(1.0-self.beta)*grad/self.beta)
        # self.iterate.mul_(self.beta)


# class ORIG_ADAM(OL_BASE):
#     def __init__(self, initial_value, **kwargs):
#         super().__init__(name='adam')
#         self.iterate = initial_value
#         self.lr = kwargs['lr']
#         self.wd = kwargs['wd']
#         self.beta2 = kwargs['beta2']
#         self.beta = kwargs['beta']
#         self.V = torch.zeros_like(initial_value)
#         self.m = torch.zeros_like(initial_value)
#         self.iterate = initial_value.clone().detach_()

#     def get_iterate(self):
#         return self.iterate


#     @torch.no_grad()
#     def update(self, grad, *args, **kwargs):
#         self.V.add_((1.0-self.beta2) * grad**2 / self.beta2)
#         self.V.mul_(self.beta2)
#         self.m.add_((1.0-self.beta) * grad / self.beta)
#         self.m.mul_(self.beta)
#         eta = self.lr * torch.rsqrt(self.V + SMALL_VALUE)
#         self.iterate.copy_(-self.lr * self.m *torch.rsqrt(self.V + SMALL_VALUE))


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
    def update(self, grad, *args, **kwargs):
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
    def update(self, grad, *args, **kwargs):
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
    'adam': ADAM,
    'origadam': ORIG_ADAM,
    'optconstada': OPTIMISTIC_CONSTRAINED_ADAGRAD,
    'constada': CONSTRAINED_ADAGRAD,
    'rewardreset': REWARD_RESET,
    'rewardresetcb': REWARD_RESET_CB,
    'rewardresetpf': REWARD_RESET_PF,
    'constl2adagrad': CONSTRAINED_L2_ADAGRAD,
    'adamda': ADAM_DA,
    'scalepf': SCALE_PF,
    'scaleadam': SCALE_ADAM,
    'scaleadamclip': SCALE_ADAM_CLIP,
    'scaleadamclipglobal': SCALE_ADAM_CLIP_GLOBAL,
    'diagscaleadam': SCALE_ADAM_DIAG,
}



class RandomOL(torch.optim.Optimizer):

    def __init__(self, params, ol='ogd', scale_type='random', logger=NoOpModule(), **kwargs):
        if 'num_lrs' not in kwargs:
            kwargs['num_lrs'] = 5
        if 'eps' not in kwargs:
            kwargs['eps'] = 0.1
        super().__init__(params, kwargs)
        self.count = 0
        self.ol = ol
        self.scale_type = scale_type
        self.logger = logger

        self.__setstate__(self.state)


    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                state['ol'] = OL_REGISTRY[self.ol](torch.zeros_like(param), **group)
                state['iterate'] = torch.clone(param).detach_()
                state['prev_iterate'] = torch.clone(param).detach_()
                state['reward'] = torch.zeros(1, device=param.device)

    @torch.no_grad()
    def swapstate(self):
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                current_value = torch.clone(param)
                param.copy_(state['iterate'])
                state['iterate'].copy_(current_value)

    @torch.no_grad()
    def swap_prev_state(self):
        for group in self.param_groups:
            for param in group['params']:
                state = self.state[param]
                current_value = torch.clone(param)
                param.copy_(state['prev_iterate'])
                state['prev_iterate'].copy_(current_value)

    @torch.no_grad()
    def resample(self):
        if self.scale_type == 'random':
            scaling = random.random()
            for group in self.param_groups:
                for param in group['params']:
                    state = self.state[param]
                    delta = state['ol'].get_iterate()
                    param.copy_(state['prev_iterate'] + scaling * delta)

    def get_inner_product(self):
        per_param_products = {}
        inner_product = 0
        for group in self.param_groups:
            wd = group['wd']
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad
                grad += wd * param
                
                state = self.state[param]
                current_product = -torch.sum(state['ol'].get_iterate() * grad)
                inner_product += current_product
                per_param_products[param] = current_product
        return inner_product, per_param_products

    @torch.no_grad()
    def step(self, closure=None):
        self.count += 1
        if self.scale_type == 'random':
            scaling = random.random()
        elif self.scale_type == 'half':
            scaling = 0.5
        elif self.scale_type == 'sgd':
            scaling = 1.0

        total_reward = 0

        inner_product, per_param_products = self.get_inner_product()

        for group in self.param_groups:
            wd = group['wd']
            for param in group['params']:
                if param.grad is None:
                    continue
                state = self.state[param]
                state['reward'].add_(per_param_products[param])
                total_reward += state['reward']

        if self.count % 100 == 0:
            self.logger.log({
                'optimizer/total_reward': total_reward,
                'optimizer/current_correlation': inner_product,
            }, commit=False)
        local_states = {}
        for group in self.param_groups:
            wd = group['wd']
            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad
                grad.add_(wd * param)


                state = self.state[param]

                local_states[param] = state['ol'].update(grad, inner_product)
            
        logging_info = {}
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]

                state['ol'].global_update(param, grad, local_states)

                delta = state['ol'].get_iterate()
                logging_info = state['ol'].get_logging_info(param, grad, local_states, logging_info)

                param.copy_(state['iterate'] + scaling * delta)
                state['prev_iterate'].copy_(state['iterate'])
                state['iterate'].add_(delta)

        if logging_info is not None and self.count % 100 == 0:
            self.logger.log(logging_info, commit=False)
        
        return 




