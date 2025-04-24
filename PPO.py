import os
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical


use_cuda = torch.cuda.is_available()
# device = torch.device('cuda' if use_cuda else 'cpu')
device = 'cpu'
print(f'using device: {device}')


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


class Memory:
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.states = []
        self.actions = []
        self.valid_action_masks = []
        self.rewards = []
        self.masks = []


    def discard_obs(self):
        self.log_probs = []
        self.values = []
        self.states = []
        self.actions = []
        self.valid_action_masks = []
        self.rewards = []
        self.masks = []


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(ActorCritic, self).__init__()

        self.num_outputs = num_outputs

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax()
        )

        self.apply(init_weights)


    def forward(self, x, valid_action_mask):
        value = self.critic(x)
        logits = self.actor(x)
        masked_logits = logits + (valid_action_mask - 1) * 1e9
        dist = Categorical(logits=masked_logits)
        return dist, value


class PPO:
    # def __init__(self, num_inputs, num_outputs, lr, agent_ids, hidden_size=128, ppo_epochs=8, mini_batch_size=2048):
    def __init__(self, num_inputs, num_outputs, lr, agent_ids, hidden_size=128, ppo_epochs=3, mini_batch_size=64, inference=False):

        self.num_inputs = num_inputs
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.hidden_size = hidden_size
        self.inference = inference

        self.model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.memories = {agent_id: Memory() for agent_id in agent_ids}
        self.state = {}
        self.dist = {}
        self.action = {}
        self.valid_action_mask = {}
        self.value = {}


    def update_learning_rate(self, learning_rate):
        print(f"Updating learning rate to: {learning_rate}")
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate


    def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, lambd=0.95):
        values_n = values + [next_value]
        values = torch.tensor(values)
        gae = 0
        advantages = torch.zeros_like(values, dtype=torch.float32)

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values_n[step + 1] * masks[step] - values_n[step]
            gae = advantages[step] = delta + gamma * lambd * masks[step] * gae

        returns = advantages + values

        return returns.to(device), advantages.to(device)


    def ppo_iter(self, mini_batch_size, states, actions, valid_action_masks, log_probs, returns, advantage):
        batch_size = states.size(0)
        rand_ids = torch.randperm(batch_size)

        for start in range(0, batch_size, mini_batch_size):
            idx = rand_ids[start: start + mini_batch_size]
            yield states[idx], actions[idx], valid_action_masks[idx], log_probs[idx], returns[idx], advantage[idx]


    def ppo_update(self, ppo_epochs, mini_batch_size, states, actions, valid_action_masks, log_probs, returns, advantages, agent_id, clip_param=0.2):
        actor_loss, critic_loss, loss = 0, 0, 0

        for _ in range(ppo_epochs):

            for state, action, valid_action_mask, old_log_probs, return_, advantage in self.ppo_iter(mini_batch_size, states, actions, valid_action_masks, log_probs, returns, advantages):
                new_dist, new_value = self.model(state, valid_action_mask)
                entropy = new_dist.entropy().mean()
                new_log_probs = new_dist.log_prob(action)

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
                actor_loss = - torch.min(surr1, surr2).mean()

                # Need to modify shape so that dimensions match
                new_value = new_value.view(-1)
                critic_loss = (return_ - new_value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss  # - 0.001 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.memories[agent_id].discard_obs()

        return actor_loss, critic_loss, loss


    def select_action(self, agent_id, state, valid_action_mask, pick_random=False):
        if not pick_random:
            with torch.no_grad():
                if not self.inference:
                    self.state[agent_id] = state
                    self.dist[agent_id], self.value[agent_id] = self.model(self.state[agent_id], valid_action_mask)
                    self.action[agent_id] = self.dist[agent_id].sample()
                    action = self.action[agent_id].cpu().item()
                    self.valid_action_mask[agent_id] = valid_action_mask
                elif self.inference:
                    dist, _ = self.model(state, valid_action_mask)
                    action = torch.argmax(dist.probs).cpu()
            return action
        else:
            valid_indices = np.where(valid_action_mask.cpu())[0]
            return np.random.choice(valid_indices)


    def store_obs(self, agent_id):
        with torch.no_grad():
            log_prob = self.dist[agent_id].log_prob(self.action[agent_id]).sum(-1)

        self.memories[agent_id].log_probs.append(log_prob.item())
        self.memories[agent_id].values.append(self.value[agent_id].item())
        self.memories[agent_id].rewards.append(0)
        self.memories[agent_id].valid_action_masks.append(self.valid_action_mask[agent_id])
        self.memories[agent_id].masks.append(1.)
        self.memories[agent_id].states.append(self.state[agent_id])
        self.memories[agent_id].actions.append(self.action[agent_id])

    
    def update_reward_done(self, agent_id, num_actions_taken, reward):
        self.memories[agent_id].rewards[-1] = reward
        self.memories[agent_id].masks[-1] = 0.


    def update(self, agent_id):
        states = torch.vstack(self.memories[agent_id].states).to(device)
        actions = torch.vstack(self.memories[agent_id].actions).to(device)
        valid_action_masks = torch.vstack(self.memories[agent_id].valid_action_masks).to(device)
        log_probs = torch.tensor(self.memories[agent_id].log_probs).to(device)

        returns, advantages = self.compute_gae(
            0,
            self.memories[agent_id].rewards,
            self.memories[agent_id].masks,
            self.memories[agent_id].values
        )

        actor_loss, critic_loss, entropy = self.ppo_update(
            self.ppo_epochs,
            self.mini_batch_size,
            states,
            actions,
            valid_action_masks,
            log_probs,
            returns,
            advantages,
            agent_id
        )

        return actor_loss, critic_loss


    def num_of_stored_obs(self, agent_id):
        return len(self.memories[agent_id].log_probs)


    def save_weights(self, filename, directory):
        torch.save(self.model.state_dict(), '%s/%s.pth' %
                   (directory, filename))


    def load(self, filename, directory):
        state_dict = torch.load('%s/%s' % (directory, filename), map_location=lambda storage, loc: storage)

        additional_dim = self.num_inputs - len(state_dict['actor.0.weight'][0])
        
        print(f'loading {filename}')
        print(f'additional inputs compared to loaded weights: {additional_dim}')

        random_weights = (torch.rand(self.hidden_size, additional_dim) - 0.5).to(device)
        state_dict['actor.0.weight'] = torch.cat((state_dict['actor.0.weight'].to(device), random_weights), 1)
        state_dict['critic.0.weight'] = torch.cat((state_dict['critic.0.weight'][:, :self.num_inputs-additional_dim].to(device), random_weights, state_dict['critic.0.weight'][:, self.num_inputs-additional_dim:].to(device)), 1)

        self.model.load_state_dict(state_dict)

    def load_pretrained_imitation(self, filename, directory):
        filepath = os.path.join(directory, filename)
        state_dict = torch.load(filepath, map_location=lambda storage, loc: storage)
        new_state_dict = {}
        new_state_dict['0.weight'] = state_dict['fc1.weight']
        new_state_dict['0.bias'] = state_dict['fc1.bias']
        new_state_dict['2.weight'] = state_dict['fc2.weight']
        new_state_dict['2.bias'] = state_dict['fc2.bias']
        new_state_dict['4.weight'] = state_dict['fc3.weight']
        new_state_dict['4.bias'] = state_dict['fc3.bias']
        new_state_dict['6.weight'] = state_dict['output.weight']
        new_state_dict['6.bias'] = state_dict['output.bias']
        self.model.actor.load_state_dict(new_state_dict)

        print(f'Pretrained weights loaded from {filepath}')