'''
model.py
implements a simple self-attention layer with some custom stuff.
'''

import torch
from torch.nn import functional as F

class ModelConfig:
    def __init__(self, vocab_size, context_length, num_layers, embedding_dim, n_heads=1, **kwargs):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_length = context_length
        self.n_heads = n_heads
        self.num_layers = num_layers
        for k,v in kwargs.items():
            setattr(self, k, v)

class SelfAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config.embedding_dim
        self.key_matrix = torch.nn.Linear(self.dim, self.dim) # maybe we want to mess with initialization schemes later?
        self.query_matrix = torch.nn.Linear(self.dim, self.dim)
        
        self.n_heads = config.n_heads
        self.context_length = config.context_length
        self.register_buffer("mask", torch.tril(torch.ones(config.context_length, config.context_length)).view(1, 1, config.context_length, config.context_length))

        assert self.dim % self.n_heads == 0, "number of heads ({}) does not evenly divide embedding dim ({})".format(self.n_heads, self.dim)


    def forward(self, x):

        # I really don't understand why heads are a good idea, but apparently they are...
        *_, T, C = x.shape

        assert C == self.dim, "specified axis does not have correct dimension: was {}, expected {}".format(C, self.dim)

        split_heads_shape = x.shape[:-1] + (self.n_heads, self.dim // self.n_heads)
        key = self.key_matrix(x).reshape(split_heads_shape).transpose(-2, -3) # [..., T, C] -> [..., T, nh, hs] -> [..., nh, T, hs]
        query = self.query_matrix(x).reshape(split_heads_shape).transpose(-2, -3) # [..., T, C] -> [..., T, nh, hs] -> [..., nh, T, hs]
        value = x.reshape(split_heads_shape).transpose(-2, -3) # [..., T, C] -> [..., T, nh, hs] -> [..., nh, T, hs]

        assert key.shape[-2] == T, "shape mismatch: {}".format(key.shape)

        print(key.shape)
        print(query.shape)

        logits = torch.matmul(key, query.transpose(-1, -2)) # [..., nh, T, hs] x [..., nh, hs, T] -> [..., nh, T, T]
        masked_logits = logits.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))

        att_weights = F.softmax(masked_logits, dim=-1)

        y = torch.matmul(att_weights, value) #  [..., nh, T, T]  x [..., nh, T, hs] -> [..., nh, T, hs]
        y = y.transpose(-2, -3).reshape(x.shape) # [..., nh, T, hs] -> [..., T, nh, hs] -> [..., T, C]

        return y





class ResidualSelfAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.selfattention = SelfAttention(config)
        self.register_parameter("residual_weight", torch.nn.Parameter(torch.tensor(1.0)))

    def forward(self, x):
        y = self.selfattention(x)
        y = x*(1-self.residual_weight) + self.residual_weight*y
        return y


class StackedAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.features = torch.nn.Sequential(*[ResidualSelfAttention(config) for _ in range(config.num_layers)])
        self.tok_embeddings = torch.nn.Embedding(config.vocab_size, config.embedding_dim)
        self.pos_embeddings = torch.nn.Parameter(torch.zeros(1, config.context_length, config.embedding_dim))
        self.head = torch.nn.Linear(config.embedding_dim, config.vocab_size)


    def get_targets(mask, idx, T):
        targets = idx[:,1:T+1]
        targets = targets.masked_fill(mask[:,1:T+1] == 0, -100)
        print(targets)
        return targets

    def forward(self, idx, mask, compute_loss=True):
        """
        idx is 1-hot encoding integer tensor shape [B, T] entries are indices into vocab
        targets is 1-hot encoding integer tensor shape [B, T], entries are indices into vocab for labels.
            ith entry of bth row of targets is label for ith prefix of idx in bth example in the batch.
        """

        # x is 1-hot encoding
        B, T = idx.size()

        T = min(T-1, config.context_length)

        tok_embd = self.tok_embeddings(idx[:, :T])

        pos_embd = self.pos_embeddings[:, :T, :]

        x = F.softmax(tok_embd + pos_embd, dim=-1) # input for attention layers: shape [B, T, C]

        features = self.features(x)

        if not compute_loss:
            return features

        logits = self.head(features) # shape [B, T, V]

        targets = StackedAttention.get_targets(mask, idx, T)

        # cross entropy loss doesn't know about T, so we flatten the time dimension:
        print("logits: ", logits.shape)
        print("targets: ", targets.shape)
        logits_for_CE = logits.reshape(-1, logits.size(-1)) # shape [BT, V]
        targets_for_CE = targets.reshape(-1) # shape [BT]

        loss = F.cross_entropy(logits_for_CE, targets_for_CE)

        return features, loss




# config = ModelConfig(vocab_size=4, context_length=3, num_layers=2, embedding_dim=4, n_heads=2)

# l = StackedAttention(config)

# idx = torch.tensor([ [2, 3, 0, 1],
#                      [0, 1, 2, 3],
#                      [3, 0, 1, 2],
#                      [1, 2, 3, 0] ])

# mask = 0*torch.tensor([[1, 1, 1, 1],
#                      [1, 1, 1, 1],
#                      [1, 1, 1, 1],
#                      [1, 1, 1, 1]])

# print(l(idx, mask)[1])

# x = torch.tensor([ [[1.0,  0.0,  0.0, 0.0],
#                     [0.0,  0.0,  0.0, 1.0],
#                     [0.0,  0.5,  0.5, 0.0]],

#                    [[0.5,  0.2,  0.1, 0.2],
#                     [0.0,  1.0,  0.0, 0.0],
#                     [0.0,  0.0,  1.0, 0.0]]])
# print(x.shape)

# y = l(x)
# print(y)

