from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from sb_env import PokerGame
import gymnasium as gym
import numpy as np

from sb3_contrib.ppo_mask import MaskablePPO


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.get_action_mask()


def agent_info(env):
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    agent_id = list(decision_steps)[0]

    return behavior_name, spec, decision_steps, terminal_steps, agent_id


def main():
    # Unity
    print('Loading Unity environment...')
    unity_env = UnityEnvironment()
    unity_env.reset()
    behavior_name, spec, decision_steps, terminal_steps, agent_id = agent_info(unity_env)

    # Python
    env = PokerGame(random_agents=True, inference=True)
    obs, info = env.reset()
    model = MaskablePPO.load('maskable_ppo_256_pretrained_mixed', env=env)

    while True:
        decision_steps, terminal_steps = unity_env.get_steps(behavior_name)
        # 1 if step the env, 0 if don't
        unity_obs = decision_steps[agent_id].obs[0]
        if unity_obs == 1:
            action_masks = env.get_action_mask()
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
        
            # Action is the list of cards, actions, player positions, winner if there is one
            unity_action = [
                *info['community_cards'],
                *info['player0_hand'],
                *info['player1_hand'],
                *info['player2_hand'],
                info['player0_last_action'],
                info['player1_last_action'],
                info['player2_last_action'],
                info['player0_position'],
                info['player1_position'],
                info['player2_position'],
                info['player0_chips'],
                info['player1_chips'],
                info['player2_chips'],
                info['player0_folded'],
                info['player1_folded'],
                info['player2_folded'],
                info['main_pot'],
                info['side_pot']
            ]

            action_array = np.array(unity_action, dtype=np.float32).reshape(1, -1)
            unity_action = ActionTuple(continuous=action_array)
            unity_env.set_actions(behavior_name, unity_action)

            unity_env.step()

            if terminated:
                obs, info = env.reset()
        else:
            unity_env.step()

if __name__ == '__main__':
    main()
