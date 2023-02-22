import os
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
from torch.nn import functional as F
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from replay_buffer import ReplayBuffer
from common import soft_update, hard_update, device
from algo.networkSD3 import Actor, Critic


class SD3:

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
        self.policy_noise = 0.1
        self.noise_clip = 0.5
        self.policy_freq = 5
        self.beta = 0.001
        self.num_noise_samples = 4
        self.with_importance_sampling = 0

        # Initialise actor network and critic network with ξ and θ
        self.actor1 = Actor(obs_dim, act_dim, num_agent, args, self.output_activation).to(self.device)
        self.actor2 = Actor(obs_dim, act_dim, num_agent, args, self.output_activation).to(self.device)
        self.critic1 = Critic(obs_dim, act_dim, num_agent, args).to(self.device)
        self.critic2 = Critic(obs_dim, act_dim, num_agent, args).to(self.device)
        self.actor1_optimizer = torch.optim.Adam(self.actor1.parameters(), lr=self.a_lr)
        self.actor2_optimizer = torch.optim.Adam(self.actor2.parameters(), lr=self.a_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=self.c_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=self.c_lr)

        # Initialise target network and critic network with ξ' ← ξ and θ' ← θ
        self.actor1_target = Actor(obs_dim, act_dim, num_agent, args, self.output_activation).to(self.device)
        self.actor2_target = Actor(obs_dim, act_dim, num_agent, args, self.output_activation).to(self.device)
        self.critic1_target = Critic(obs_dim, act_dim, num_agent, args).to(self.device)
        self.critic2_target = Critic(obs_dim, act_dim, num_agent, args).to(self.device)
        hard_update(self.actor1, self.actor1_target)
        hard_update(self.critic1, self.critic1_target)
        hard_update(self.actor2, self.actor2_target)
        hard_update(self.critic2, self.critic2_target)

        # Initialise replay buffer R
        self.replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size)

        self.c1_loss = None
        self.a1_loss = None
        self.c2_loss = None
        self.a2_loss = None

    # Random process N using epsilon greedy
    def choose_action(self, obs, evaluation=False):
        obs = torch.Tensor(obs).to(self.device)
        
        action1 = self.actor1(obs).cpu().detach().numpy()
        action2 = self.actor2(obs).cpu().detach().numpy()

        action1_ = self.actor1(obs)
        action2_ = self.actor2(obs)
        
        q1 = self.critic1(obs, action1_)
        q2 = self.critic2(obs, action2_)

        action = action1 if torch.sum(q1) >= torch.sum(q2) else action2

        return action

    def update1(self):

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
            next_action = self.actor1_target(next_state_batch)                
            noise = torch.randn(
                (action_batch.shape[0], action_batch.shape[1], self.num_noise_samples), 
                dtype=action_batch.dtype, layout=action_batch.layout, device=action_batch.device
            )
            
            noise = noise * self.policy_noise
			
            noise_pdf = self.calc_pdf(noise) if self.with_importance_sampling else None
			
            noise = noise.clamp(0, 2*self.noise_clip)
            
            # next_action = torch.unsqueeze(next_action, 1)
            # print(next_action.shape,noise.shape)

            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # next_state = torch.unsqueeze(next_state_batch, 1)
            # next_state = next_state.repeat((1, self.num_noise_samples, 1))

            next_Q1 = self.critic1_target(next_state_batch, next_action)
            next_Q2 = self.critic2_target(next_state_batch, next_action)
            
            next_Q = torch.min(next_Q1, next_Q2)
            # next_Q = torch.squeeze(next_Q, 2)

            # softmax_next_Q = self.softmax_operator(next_Q, noise_pdf)
            next_Q = F.softmax(next_Q, 1)

            target_Q = reward_batch + (1-done_batch) * self.gamma * next_Q

        current_Q = self.critic1(state_batch, action_batch)

        critic1_loss = F.mse_loss(current_Q, target_Q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        actor1_loss = -self.critic1(state_batch, self.actor1(state_batch)).mean()
			
        self.actor1_optimizer.zero_grad()
        actor1_loss.backward()
        self.actor1_optimizer.step()

        self.c1_loss = critic1_loss.item()
        self.a1_loss = actor1_loss.item()

        # Update the target networks
        soft_update(self.actor1, self.actor1_target, self.tau)
        soft_update(self.critic1, self.critic1_target, self.tau)

        return self.c1_loss, self.a1_loss
    
    def update2(self):

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
            next_action = self.actor2_target(next_state_batch)                
            noise = torch.randn(
                (action_batch.shape[0], action_batch.shape[1], self.num_noise_samples), 
                dtype=action_batch.dtype, layout=action_batch.layout, device=action_batch.device
            )
            
            noise = noise * self.policy_noise
			
            noise_pdf = self.calc_pdf(noise) if self.with_importance_sampling else None
			
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            
            # next_action = torch.unsqueeze(next_action, 1)
            # print(next_action.shape,noise.shape)

            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # next_state = torch.unsqueeze(next_state_batch, 1)
            # next_state = next_state.repeat((1, self.num_noise_samples, 1))

            next_Q1 = self.critic1_target(next_state_batch, next_action)
            next_Q2 = self.critic2_target(next_state_batch, next_action)
            
            next_Q = torch.min(next_Q1, next_Q2)
            # next_Q = torch.squeeze(next_Q, 2)
            
            # softmax_next_Q = self.softmax_operator(next_Q, noise_pdf)
            next_Q = F.softmax(next_Q, 1)

            target_Q = reward_batch + (1-done_batch) * self.gamma * next_Q

        current_Q = self.critic2(state_batch, action_batch)

        critic2_loss = F.mse_loss(current_Q, target_Q)

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        actor2_loss = -self.critic2(state_batch, self.actor2(state_batch)).mean()
			
        self.actor2_optimizer.zero_grad()
        actor2_loss.backward()
        self.actor2_optimizer.step()

        self.c2_loss = critic2_loss.item()
        self.a2_loss = actor2_loss.item()

        # Update the target networks
        soft_update(self.actor2, self.actor2_target, self.tau)
        soft_update(self.critic2, self.critic2_target, self.tau)

        return self.c2_loss, self.a2_loss
    
    def update(self):
        c_1loss, a1_loss = self.update1()
        c_2loss, a2_loss = self.update2()
        return c_1loss, a1_loss, c_2loss, a2_loss

    def calc_pdf(self, samples, mu=0):
        pdfs = 1/(self.policy_noise * np.sqrt(2 * np.pi)) * torch.exp( - (samples - mu)**2 / (2 * self.policy_noise**2) )
        pdf = torch.prod(pdfs, dim=2)
        return pdf
    
    def softmax_operator(self, q_vals, noise_pdf=None):
        max_q_vals = torch.max(q_vals, 1, keepdim=True).values
        norm_q_vals = q_vals - max_q_vals
        e_beta_normQ = torch.exp(self.beta * norm_q_vals)
        Q_mult_e = q_vals * e_beta_normQ

        numerators = Q_mult_e
        denominators = e_beta_normQ

        if self.with_importance_sampling:
            numerators /= noise_pdf
            denominators /= noise_pdf

        sum_numerators = torch.sum(numerators, 1)
        sum_denominators = torch.sum(denominators, 1)

        # softmax_q_vals = sum_numerators / sum_denominators
        softmax_q_vals = numerators / denominators

        # softmax_q_vals = torch.unsqueeze(softmax_q_vals, 1)
        return softmax_q_vals

    def get_loss(self):
        return self.c1_loss, self.a1_loss, self.c2_loss, self.a2_loss

    def load_model(self, run_dir, episode):
        print(f'\nBegin to load model: ')
        base_path = os.path.join(run_dir, 'trained_model')
        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        print(f'Actor path: {model_actor_path}')
        print(f'Critic path: {model_critic_path}')

        if os.path.exists(model_critic_path) and os.path.exists(model_actor_path):
            actor = torch.load(model_actor_path, map_location=device)
            critic = torch.load(model_critic_path, map_location=device)
            self.actor.load_state_dict(actor)
            self.critic.load_state_dict(critic)
            print("Model loaded!")
        else:
            sys.exit(f'Model not founded!')

    def save_model(self, run_dir, episode):
        base_path = os.path.join(run_dir, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "actor1_" + str(episode) + ".pth")
        torch.save(self.actor1.state_dict(), model_actor_path)
        model_actor_path = os.path.join(base_path, "actor2_" + str(episode) + ".pth")
        torch.save(self.actor2.state_dict(), model_actor_path)

        model_critic_path = os.path.join(base_path, "critic1_" + str(episode) + ".pth")
        torch.save(self.critic1.state_dict(), model_critic_path)
        model_critic_path = os.path.join(base_path, "critic2_" + str(episode) + ".pth")
        torch.save(self.critic2.state_dict(), model_critic_path)
