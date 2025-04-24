from sb_env import PokerGame
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

from expert_agent import MLP


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.get_action_mask()


# expert_agent = MLP(input_size=19, num_classes=5, act='tanh')
# expert_agent.load_state_dict(torch.load('imitation_learning/pretrained_tanh.pth'))
# expert_agent.eval()

# state_dict = expert_agent.state_dict()
# new_state_dict = {}
# new_state_dict['0.weight'] = state_dict['fc1.weight']
# new_state_dict['0.bias'] = state_dict['fc1.bias']
# new_state_dict['2.weight'] = state_dict['fc2.weight']
# new_state_dict['2.bias'] = state_dict['fc2.bias']
# new_state_dict['4.weight'] = state_dict['fc3.weight']
# new_state_dict['4.bias'] = state_dict['fc3.bias']
# new_state_dict['6.weight'] = state_dict['output.weight']
# new_state_dict['6.bias'] = state_dict['output.bias']


# env = PokerGame(random_agents=True, inference=True)
# env = ActionMasker(env, mask_fn)  # Wrap to enable masking


# model = MaskablePPO(
#     MaskableActorCriticPolicy,  # Policy model
#     env,  # Environment
#     n_steps=2048,  # Number of steps per update
#     batch_size=32,  # Minibatch size
#     n_epochs=3,  # Number of epochs per training step
#     policy_kwargs=dict(net_arch=[256, 256, 256], activation_fn=nn.Tanh),  # Neural network architecture
#     tensorboard_log="./poker_maskable_ppo_tensorboard/",  # Log training progress
#     verbose=1  # Print training logs
# )

# checkpoint = torch.load('policy_tanh.pth', map_location="cpu")
# model.policy.load_state_dict(checkpoint, strict=False)

# model = MaskablePPO.load('maskable_ppo_poker_final_256_pretrained_mixed_allin_tanh', env=env)

# ppo_actor = model.policy.mlp_extractor.policy_net
# ppo_actor.load_state_dict(new_state_dict, strict=False)

# with torch.no_grad():
#     model.policy.action_net.weight.copy_(expert_agent.output.weight)
#     model.policy.action_net.bias.copy_(expert_agent.output.bias)

# model.learn(4000000)
# model.save("maskable_ppo_poker_final_256_pretrained_mixed_allin_tanh")

# print(evaluate_policy(model, env, n_eval_episodes=100000, warn=False))

env = PokerGame(random_agents=True, inference=True)
expert_agent = MLP(input_size=19, num_classes=5, act='tanh')
expert_agent.load_state_dict(torch.load('imitation_learning/pretrained_tanh.pth'))
expert_agent.eval()

obs, _ = env.reset()
model = MaskablePPO.load('maskable_ppo_256_pretrained_mixed', env=env)

eps = 0
while eps < 50000:
    action_masks = env.get_action_mask()
    # action = expert_agent.select_action(obs, action_masks)

    # valid_indices = np.where(action_masks)[0]
    # action = np.random.choice(valid_indices)

    action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        obs, _ = env.reset()
        eps += 1