#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from config.config import DEVICE, SUMMARY_WRITER_RUN_LOG
from .actor import Actor
from .critic import Critic

class td3(object):
    def __init__(self, state_dim, action_dim, max_action, env,
                 history_size, model_dim=256, nhead=4, num_encoder_layers=2):
       
        # Initialize the Actor network
        self.actor = Actor(
            state_dim, action_dim, history_size, model_dim, nhead, num_encoder_layers
        ).to(DEVICE)
        self.actor_target = Actor(
            state_dim, action_dim, history_size, model_dim, nhead, num_encoder_layers
        ).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        # Initialize the Critic networks
        self.critic = Critic(
            state_dim, action_dim, history_size, model_dim, nhead, num_encoder_layers
        ).to(DEVICE)
        self.critic_target = Critic(
            state_dim, action_dim, history_size, model_dim, nhead, num_encoder_layers
        ).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.writer = SummaryWriter(log_dir=SUMMARY_WRITER_RUN_LOG)
        self.iter_count = 0
        self.env = env

    def get_action(self, state, history):
        state_tensor = torch.Tensor(state.reshape(1, -1)).to(DEVICE)
        history_tensor = torch.Tensor(history.reshape(1, -1, state.shape[0])).to(DEVICE)
        
        return self.actor(state_tensor, history_tensor).cpu().data.numpy().flatten()

    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=16,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    ):
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        
        for it in range(iterations):
            (
                state,
                action,
                reward,
                done,
                next_state,
                history,
                next_history
            ) = replay_buffer.sample_batch(batch_size)
            
            
            with torch.no_grad():
                # Obtain the estimated action from the next state by using the actor-target
                next_action = self.actor_target(next_state, next_history)
                 # Add noise to the action
                noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

                # Calculate the Q values from the critic-target network for the next state-action pair and next_history
                target_Q1, target_Q2 = self.critic_target(next_state, next_action, next_history)
                target_Q = torch.min(target_Q1, target_Q2)
                av_Q += torch.mean(target_Q)
                max_Q = max(max_Q, torch.max(target_Q))
                
                # Calculate the final Q value from the target network parameters by using Bellman equation
                target_Q = reward + ((1 - done) * discount * target_Q)

            # Get the Q values of the basis networks with the current parameters
            current_Q1, current_Q2 = self.critic(state, action, history)

            # Calculate Critic-Loss between the current Q value and the target Q value
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Perform the gradient descent and update
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values
                # (essentially perform gradient ascent)t
                actor_action = self.actor(state, history)
                actor_loss = -self.critic.Q1(state, actor_action, history).mean()

                # Actor-Update
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Use soft update to update the actor-target network parameters by
                # infusing small amount of current parameters
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                # Use soft update to update the critic-target network parameters by infusing
                # small amount of current parameters
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            av_loss += loss.item()
            
        self.iter_count += 1
        avg_loss_val = av_loss / iterations
        avg_q_val = av_Q / iterations
        
        self.env.get_logger().info(f"writing new results for a tensorboard")
        self.env.get_logger().info(f"Loss: {avg_loss_val:.3f}, Av.Q: {avg_q_val:.3f}, Max.Q: {max_Q:.3f}, Iter: {self.iter_count}")
        self.writer.add_scalar("loss", avg_loss_val, self.iter_count)
        self.writer.add_scalar("Av. Q", avg_q_val, self.iter_count)
        self.writer.add_scalar("Max. Q", max_Q, self.iter_count)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth", map_location=DEVICE))
        self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth", map_location=DEVICE))

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())