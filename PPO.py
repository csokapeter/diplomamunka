import torch
import torch.nn as nn
from torch.distributions import Normal


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print(f'using device: {device}')


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)


class Memory:
    def __init__(self):
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.states: list[torch.tensor] = []
        self.actions: list[torch.tensor] = []
        self.rewards: list[float] = []
        self.masks: list[float] = []


    def discard_obs(self, val):
        self.log_probs = self.log_probs[-val:]
        self.values = self.values[-val:]
        self.states = self.states[-val:]
        self.actions = self.actions[-val:]
        self.rewards = self.rewards[-val:]
        self.masks = self.masks[-val:]


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, action_std):
        super(ActorCritic, self).__init__()

        self.num_outputs = num_outputs

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Tanh()
        )

        # self.action_std = torch.tensor([action_std, action_std])
        self.log_action_std = nn.Parameter(torch.zeros(num_outputs))
        self.apply(init_weights)


    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)
        log_action_std = self.log_action_std.expand_as(mu)
        std = torch.exp(log_action_std)
        # std = self.action_std.expand_as(mu).to(device)
        dist = Normal(mu, std)
        return dist, value


class PPO:
    def __init__(self, num_inputs, num_outputs, lr, action_std, agent_ids, hidden_size=128, ppo_epochs=8, mini_batch_size=2048):

        self.num_inputs = num_inputs
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.hidden_size = hidden_size

        self.model = ActorCritic(num_inputs, num_outputs, hidden_size, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.agent_memories = {agent_id: Memory() for agent_id in agent_ids}
        self.state = {}
        self.dist = {}
        self.action = {}
        self.value = {}


    def update_learning_rate(self, learning_rate):
        print(f"Updating learning rate to: {learning_rate}")
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate


    def set_action_std(self, action_std):
        print(f'Updating standard deviation to: {self.model.action_std}')
        self.model.action_std = torch.tensor([action_std, action_std])


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


    def ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        rand_ids = torch.randperm(batch_size)

        for start in range(0, batch_size, mini_batch_size):
            idx = rand_ids[start: start + mini_batch_size]
            yield states[idx], actions[idx], log_probs[idx], returns[idx], advantage[idx]


    def ppo_update(self, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, agent_id, clip_param=0.2):
        actor_loss, critic_loss, loss = 0, 0, 0

        for _ in range(ppo_epochs):

            for state, action, old_log_probs, return_, advantage in self.ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
                new_dist, new_value = self.model(state)
                entropy = new_dist.entropy().mean()
                new_log_probs = new_dist.log_prob(action).sum(-1)

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

        self.agent_memories[agent_id].discard_obs((len(self.agent_memories[agent_id].log_probs) // 4) * 3)

        return actor_loss, critic_loss, loss


    def select_action(self, agent_id, state):
        with torch.no_grad():
            self.state[agent_id] = state
            self.dist[agent_id], self.value[agent_id] = self.model(self.state[agent_id])
            self.action[agent_id] = self.dist[agent_id].sample()

        return self.action[agent_id].cpu().numpy()


    def store_obs(self, agent_id, reward, done):
        with torch.no_grad():
            log_prob = self.dist[agent_id].log_prob(self.action[agent_id]).sum(-1)

        self.agent_memories[agent_id].log_probs.append(log_prob.item())
        self.agent_memories[agent_id].values.append(self.value[agent_id].item())
        self.agent_memories[agent_id].rewards.append(reward)
        self.agent_memories[agent_id].masks.append(float(1 - done))
        self.agent_memories[agent_id].states.append(self.state[agent_id])
        self.agent_memories[agent_id].actions.append(self.action[agent_id])


    def num_of_stored_obs(self, agent_id):
        return len(self.agent_memories[agent_id].log_probs)


    def update(self, agent_id, next_state):
        states = torch.vstack(self.agent_memories[agent_id].states).to(device)
        actions = torch.vstack(self.agent_memories[agent_id].actions).to(device)
        log_probs = torch.tensor(self.agent_memories[agent_id].log_probs).to(device)

        with torch.no_grad():
            _, next_value = self.model(next_state)
            next_value = next_value.item()

        returns, advantages = self.compute_gae(
            next_value,
            self.agent_memories[agent_id].rewards,
            self.agent_memories[agent_id].masks,
            self.agent_memories[agent_id].values
        )

        actor_loss, critic_loss, entropy = self.ppo_update(
            self.ppo_epochs,
            self.mini_batch_size,
            states,
            actions,
            log_probs,
            returns,
            advantages,
            agent_id
        )


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
        # state_dict['log_action_std'] = torch.zeros(2).to(device)

        self.model.load_state_dict(state_dict)
