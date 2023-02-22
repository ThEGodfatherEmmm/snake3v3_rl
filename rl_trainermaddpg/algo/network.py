from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from common import *
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_SIZE = 256


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, args, output_activation='tanh'):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = 1

        self.args = args

        self.linear_a1 = nn.Linear(obs_dim, HIDDEN_SIZE)
        self.linear_a2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear_a = nn.Linear(HIDDEN_SIZE, act_dim)
        
        # Activation func init
        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh= nn.Tanh()
        self.train()

    def forward(self, obs_batch):
        x = self.LReLU(self.linear_a1(obs_batch))
        x = self.LReLU(self.linear_a2(x))
        policy = self.tanh(self.linear_a(x))
        return policy 


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, args):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = 3

        self.args = args

        sizes_prev = [obs_dim + act_dim, HIDDEN_SIZE]

        sizes_post = [HIDDEN_SIZE, HIDDEN_SIZE, 1]

        self.prev_dense = mlp(sizes_prev)
        self.post_dense = mlp(sizes_post)

    def forward(self, obs_batch, action_batch):
        out = torch.cat((obs_batch, action_batch), dim=-1)
        out = self.prev_dense(out)
        out = self.post_dense(out)
        return out



