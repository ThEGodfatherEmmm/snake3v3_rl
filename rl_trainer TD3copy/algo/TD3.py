import os
import torch
import copy
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from replay_buffer import ReplayBuffer
from common import soft_update, hard_update, device
from algo.networkTD3 import Actor, Critic


class TD3:

    def __init__(self, obs_dim, act_dim, num_agent, args):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agent = num_agent
        self.device = device
        self.a_lr = args.a_lr
        self.c_lr = args.c_lr
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.model_episode = args.model_episode
        self.eps = args.epsilon
        self.decay_speed = args.epsilon_speed
        self.output_activation = args.output_activation
        self.max_action = 1
        self.total_it = 0
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 5

        # # Initialise actor network and critic network with ξ and θ
        # self.actor = Actor(obs_dim, act_dim, num_agent, args, self.output_activation).to(self.device)
        # self.critic = Critic(obs_dim, act_dim, num_agent, args).to(self.device)
        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)

        # # Initialise target network and critic network with ξ' ← ξ and θ' ← θ
        # self.actor_target = Actor(obs_dim, act_dim, num_agent, args, self.output_activation).to(self.device)
        # self.critic_target = Critic(obs_dim, act_dim, num_agent, args).to(self.device)
        # hard_update(self.actor, self.actor_target)
        # hard_update(self.critic, self.critic_target)
        self.actor = Actor(obs_dim, act_dim, num_agent, args, self.output_activation).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        #定义critic网络
        self.critic1 = Critic(obs_dim, act_dim, num_agent, args).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2 = Critic(obs_dim, act_dim, num_agent, args).to(self.device)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=3e-4)
        # Initialise replay buffer R
        self.replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size)

        self.c_loss = None
        self.a_loss = None

    # Random process N using epsilon greedy
    def choose_action(self, obs, evaluation=False):

        p = np.random.random()
        if p > self.eps or evaluation:
            obs = torch.Tensor([obs]).to(self.device)
            action = self.actor(obs).cpu().detach().numpy()[0]
        else:
            action = self.random_action()
        self.eps *= self.decay_speed
        return action

    def random_action(self):
        if self.output_activation == 'tanh':
            return np.random.uniform(low=-1, high=1, size=(self.num_agent, self.act_dim))
        return np.random.uniform(low=0, high=1, size=(self.num_agent, self.act_dim))

    def update(self):
        self.total_it += 1
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        # Sample a greedy_min mini-batch of M transitions from R
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.get_batches()

        state_batch = torch.Tensor(state_batch).reshape(self.batch_size, self.num_agent, -1).to(self.device)
        action_batch = torch.Tensor(action_batch).reshape(self.batch_size, self.num_agent, -1).to(self.device)
        reward_batch = torch.Tensor(reward_batch).reshape(self.batch_size, self.num_agent, 1).to(self.device)
        next_state_batch = torch.Tensor(next_state_batch).reshape(self.batch_size, self.num_agent, -1).to(self.device)
        done_batch = torch.Tensor(done_batch).reshape(self.batch_size, self.num_agent, 1).to(self.device)

        # Compute target value for each agents in each transition using the Bi-RNN
        with torch.no_grad():
            noise = (torch.rand_like(action_batch) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state_batch) + noise).clamp(-self.max_action, self.max_action)
            target_Q1 = self.critic1_target(next_state_batch, next_action)
            target_Q2 = self.critic2_target(next_state_batch, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward_batch + (1-done_batch) * self.gamma * target_Q

        # Compute critic gradient estimation according to Eq.(8)
        current_Q1 = self.critic1(state_batch, action_batch)
        current_Q2 = self.critic2(state_batch, action_batch)
        loss_Q1 = F.mse_loss(current_Q1, target_Q)
        loss_Q2 = F.mse_loss(current_Q2, target_Q)
        critic_loss =  loss_Q1 + loss_Q2

        # Update the critic networks based on Adam
        self.critic1_optimizer.zero_grad()
        clip_grad_norm_(self.critic1.parameters(), 1)
        loss_Q1.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        clip_grad_norm_(self.critic2.parameters(), 1)
        loss_Q2.backward()
        self.critic2_optimizer.step()

        # Compute actor gradient estimation according to Eq.(7)
        # and replace Q-value with the critic estimation
        if (self.total_it % self.policy_freq == 0) or (self.total_it < 2):

            q1 = self.critic1(state_batch, self.actor(state_batch))
            q2 = self.critic2(state_batch, self.actor(state_batch))
            actor_loss = -torch.min(q1, q2).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            clip_grad_norm_(self.actor.parameters(), 1)
            self.actor_optimizer.step()
            self.c_loss = critic_loss.item()
            self.a_loss = actor_loss.item()


        # Update the target networks
        soft_update(self.actor, self.actor_target, self.tau)
        soft_update(self.critic1, self.critic1_target, self.tau)
        soft_update(self.critic2, self.critic2_target, self.tau)

        return self.c_loss, self.a_loss

    def get_loss(self):
        return self.c_loss, self.a_loss

    def load_model(self, run_dir, episode):
        print(f'\nBegin to load model: ')
        base_path = os.path.join(run_dir, 'trained_model')
        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        model_critic1_path = os.path.join(base_path, "critic1_" + str(episode) + ".pth")
        model_critic2_path = os.path.join(base_path, "critic2_" + str(episode) + ".pth")
        print(f'Actor path: {model_actor_path}')
        print(f'Critic path: {model_critic1_path}')
        print(f'Critic path: {model_critic2_path}')

        if os.path.exists(model_critic1_path) and os.path.exists(model_actor_path):
            actor = torch.load(model_actor_path, map_location=device)
            critic1 = torch.load(model_critic1_path, map_location=device)
            critic2 = torch.load(model_critic2_path, map_location=device)
            self.actor.load_state_dict(actor)
            self.critic1.load_state_dict(critic1)
            self.critic2.load_state_dict(critic2)
            print("Model loaded!")
        else:
            sys.exit(f'Model not founded!')

    def save_model(self, run_dir, episode):
        base_path = os.path.join(run_dir, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        torch.save(self.actor.state_dict(), model_actor_path)

        model_critic1_path = os.path.join(base_path, "critic1_" + str(episode) + ".pth")
        torch.save(self.critic1.state_dict(), model_critic1_path)
        model_critic2_path = os.path.join(base_path, "critic2_" + str(episode) + ".pth")
        torch.save(self.critic2.state_dict(), model_critic2_path)