import os
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import date
from phevaluator.evaluator import evaluate_cards
import multiprocessing as mp

from PPO import PPO
from utils import *
from card import Card, Deck
from pot import Pot
from player import Player


use_cuda = torch.cuda.is_available()
# device = torch.device('cuda' if use_cuda else 'cpu')
device = 'cpu'


def test_agent(agent, agent_ids, num_games):
    # print('Testing agent winrate...')
    num_players = 3
    wins = 0
    
    for i in range(num_games):
        starting_chips = 150
        players = [Player(starting_chips, agent_ids[i]) for i in range(num_players)]
        players[1].takes_random_actions = True
        players[2].takes_random_actions = True
        designated_player = players[0]
        dealer_position = 0
        num_blind_posted = 1
        num_blind_increase = 0
        small_blind = 5

        while len(players) > 1:
            deck = Deck()
            is_heads_up = len(players) == 2
            if is_heads_up:
                sb_position = dealer_position
                bb_position = (dealer_position + 1) % 2
            else:
                sb_position = (dealer_position + 1) % len(players)
                bb_position = (dealer_position + 2) % len(players)

            if num_blind_posted % 5 == 0:
                small_blind *= 2
                num_blind_increase += 1
            players[sb_position].bet(small_blind)
            players[bb_position].bet(2 * small_blind)
            num_blind_posted += 1

            players[dealer_position].position = 0
            players[sb_position].position = 1
            players[bb_position].position = 2

            # Dealing initial hands
            for player in players:
                player.hand = deck.deal(2)
            
            pot = Pot()
            community_cards = []
            betting_round(agent, players, community_cards, pot, big_blind=small_blind*2, num_blind_increase=num_blind_increase, street=0, current_highest_bet=2*small_blind, starting_player=dealer_position)

            first_act_pos = bb_position if players[sb_position].is_all_in or players[sb_position].folded else sb_position

            community_cards += deck.deal(3)
            betting_round(agent, players, community_cards, pot, big_blind=small_blind*2, num_blind_increase=num_blind_increase, street=1, current_highest_bet=0, starting_player=first_act_pos)

            first_act_pos = bb_position if players[sb_position].is_all_in or players[sb_position].folded else sb_position

            community_cards += deck.deal(1)
            betting_round(agent, players, community_cards, pot, big_blind=small_blind*2, num_blind_increase=num_blind_increase, street=2, current_highest_bet=0, starting_player=first_act_pos)

            first_act_pos = bb_position if players[sb_position].is_all_in or players[sb_position].folded else sb_position

            community_cards += deck.deal(1)
            betting_round(agent, players, community_cards, pot, big_blind=small_blind*2, num_blind_increase=num_blind_increase, street=3, current_highest_bet=0, starting_player=first_act_pos)

            main_pot_contenders = list(pot.main_pot_contributors)
            main_pot_winners = [main_pot_contenders[0]]
            best_eval = evaluate_cards(*[str(e) for e in main_pot_contenders[0].hand + community_cards])
            for i in range(1, len(main_pot_contenders)):
                hand_eval = evaluate_cards(*[str(e) for e in main_pot_contenders[i].hand + community_cards])
                if hand_eval < best_eval:
                    main_pot_winners = [main_pot_contenders[i]]
                    best_eval = hand_eval
                elif hand_eval == best_eval:
                    main_pot_winners.append(main_pot_contenders[i])
            for p in main_pot_winners:
                p.chips += pot.main_pot // len(main_pot_winners)

            if pot.side_pot_contributors:
                side_pot_contenders = list(pot.side_pot_contributors)
                side_pot_winners = [side_pot_contenders[0]]
                best_eval = evaluate_cards(*[str(e) for e in side_pot_contenders[0].hand + community_cards])
                for i in range(1, len(side_pot_contenders)):
                    hand_eval = evaluate_cards(*[str(e) for e in side_pot_contenders[i].hand + community_cards])
                    if hand_eval < best_eval:
                        side_pot_winners = [side_pot_contenders[i]]
                        best_eval = hand_eval
                    elif hand_eval == best_eval:
                        side_pot_winners.append(side_pot_contenders[i])
                for p in side_pot_winners:
                    p.chips += pot.side_pot // len(side_pot_winners)

            for p in players:
                p.folded = False
                p.is_all_in = False
                p.actions_taken = 0
                p.current_bet = 0
                p.last_action = -1
                p.num_actions_this_episode = 0
                p.chips_at_start_of_ep = p.chips
            
            players = [p for p in players if p.chips != 0]
            dealer_position = (dealer_position + 1) % len(players)
            if len(players) == 1 and players[0] == designated_player:
                wins += 1

    return wins / num_games


# def get_player_action(valid_actions):
#     action = None
#     while action not in valid_actions:
#         action = input(f'choose {", ".join(valid_actions)}\n')
#     return action

# Actions:
# 0 - fold
# 1 - check
# 2 - call
# 3 - 2 BB raise
# 4 - 4 BB raise
# 5 - all in

def betting_round(agent, players, community_cards, pot, big_blind, num_blind_increase, street, current_highest_bet, starting_player=0):
    num_players = len(players)
    current_player = starting_player
    pot.main_pot_cutoff = current_highest_bet
    pot.side_pot_cutoff = 0
    t = -1

    for p in players:
        if p.is_all_in:
            p.actions_taken = 1
            pot.main_pot_cutoff = p.current_bet
            pot.side_pot_cutoff = current_highest_bet - pot.main_pot_cutoff
            is_sidepot = True

    while not all((p.current_bet == current_highest_bet or p.is_all_in or p.folded) and p.actions_taken > 0 for p in players):
        p = players[current_player]
        all_in_players = [player for player in players if player.is_all_in]
        is_sidepot = len(all_in_players) > 0

        active_players = [player for player in players if not player.folded and not player.is_all_in]
        if len(active_players) == 1 and p.current_bet == current_highest_bet:
            p.actions_taken += 1
            current_player = (current_player + 1) % num_players
            continue

        if not p.folded and not p.is_all_in:
            # print(p.hand, p.chips)
            obs = get_obs(p.hand, community_cards, num_blind_increase, street, p, players)
            # print(obs)

            # Determine if all other players are all-in or folded
            all_other_players_all_in_or_folded = len(active_players) == 1

            # Check if the current player has to go all-in to call
            must_go_all_in = p.chips <= (current_highest_bet - p.current_bet)

            # If player has to call all-in, or all others are all-in/folded, restrict options to "call" or "fold"
            if must_go_all_in or all_other_players_all_in_or_folded:
                # action = get_player_action(['call', 'fold'])
                valid_action_mask = torch.tensor([1., 0., 1., 0., 0.]).to(device)
                action = agent.select_action(p.id, torch.tensor(obs, dtype=torch.float32).to(device), valid_action_mask, p.takes_random_actions)
                p.last_action = action
                # print(action)
                if action == 0:
                    p.fold()
                    if p in pot.main_pot_contributors:
                        pot.main_pot_contributors.remove(p)
                    if p in pot.side_pot_contributors:
                        pot.side_pot_contributors.remove(p)
                elif action == 2:
                    # p.call(pot.main_pot_cutoff + pot.side_pot_cutoff)
                    p.call(current_highest_bet)
                    if is_sidepot and p.current_bet < pot.side_pot_cutoff:
                        pot.side_pot_cutoff = p.current_bet
                    elif p.current_bet < pot.main_pot_cutoff:
                        pot.side_pot_cutoff = pot.main_pot_cutoff - p.current_bet
                        pot.main_pot_cutoff = p.current_bet
                        
            else:
                if p.current_bet < current_highest_bet:
                    t = 0
                    # action = get_player_action(['call', 'raise', 'fold'])
                    valid_action_mask = torch.tensor([1., 0., 1., 1., 1.]).to(device)
                    action = agent.select_action(p.id, torch.tensor(obs, dtype=torch.float32).to(device), valid_action_mask, p.takes_random_actions)
                    p.last_action = action
                    if action == 0:
                        p.fold()
                        if p in pot.main_pot_contributors:
                            pot.main_pot_contributors.remove(p)
                        if p in pot.side_pot_contributors:
                            pot.side_pot_contributors.remove(p)
                    elif action == 2:
                        p.call(pot.main_pot_cutoff + pot.side_pot_cutoff)
                    elif action > 2:
                        if action == 3:
                            raise_amount = current_highest_bet + (2 - (current_highest_bet == big_blind)) * big_blind
                        # elif action == 4:
                        #     raise_amount = current_highest_bet + 4 * big_blind
                        elif action == 4:
                            raise_amount = current_highest_bet + 15 * big_blind
                        actual_raise_amount = p.raise_bet(raise_amount, current_highest_bet)
                        current_highest_bet += actual_raise_amount
                        if is_sidepot:
                            pot.side_pot_cutoff += actual_raise_amount
                        else:
                            pot.main_pot_cutoff += actual_raise_amount

                else:
                    t = 1
                    # action = get_player_action(['check', 'raise'])
                    valid_action_mask = torch.tensor([0., 1., 0., 1., 1.]).to(device)
                    action = agent.select_action(p.id, torch.tensor(obs, dtype=torch.float32).to(device), valid_action_mask, p.takes_random_actions)
                    p.last_action = action
                    if action > 2:
                        if action == 3:
                            raise_amount = current_highest_bet + (2 - (current_highest_bet == big_blind)) * big_blind
                        # elif action == 4:
                        #     raise_amount = current_highest_bet + 4 * big_blind
                        elif action == 4:
                            raise_amount = current_highest_bet + 15 * big_blind
                        actual_raise_amount = p.raise_bet(raise_amount, current_highest_bet)
                        current_highest_bet += actual_raise_amount
                        if is_sidepot:
                            pot.side_pot_cutoff += actual_raise_amount
                        else:
                            pot.main_pot_cutoff += actual_raise_amount

            p.actions_taken += 1
            if p.designated_agent:
                agent.store_obs(p.id)
            # print(f'{action =}')
            p.num_actions_this_episode += 1
            if p.num_actions_this_episode > 50:
                print(p.current_bet, current_highest_bet, p.is_all_in, p.folded, p.actions_taken, t)
                print('NUM ACT IS BIGGER THAN 50 HELP')
                break
            # input('q')

        current_player = (current_player + 1) % num_players
        # for p in players:
        #     print(p.hand, p.folded, p.current_bet, p.actions_taken, current_highest_bet)
        #     print()

    all_in_players = [player for player in players if player.is_all_in]
    is_sidepot = len(all_in_players) > 0
    
    # Iterate over all players and properly allocate their contributions
    for p in players:
        if p.folded:
            # A folded player's bet should go to the main pot or side pot
            if not is_sidepot:  # No side pot, everything goes to the main pot
                pot.main_pot += p.current_bet
            else:  # Side pot exists, so allocate to side pot if necessary
                if p.current_bet <= pot.main_pot_cutoff:
                    pot.main_pot += p.current_bet
                else:
                    # If they contributed more than the main pot cutoff, allocate excess to side pot
                    pot.main_pot += pot.main_pot_cutoff
                    pot.side_pot += p.current_bet - pot.main_pot_cutoff

        elif p.is_all_in:
            # All-in player's bet should be split between the main and side pots
            if p.current_bet <= pot.main_pot_cutoff:
                pot.main_pot += p.current_bet
                pot.main_pot_contributors.add(p)
            else:
                pot.main_pot += pot.main_pot_cutoff
                pot.side_pot += p.current_bet - pot.main_pot_cutoff
                pot.main_pot_contributors.add(p)
                pot.side_pot_contributors.add(p)

        else:  # Active player who hasn't folded or gone all-in
            # Allocate to the main pot or side pot based on their current bet
            if p.current_bet >= pot.main_pot_cutoff:
                pot.main_pot += pot.main_pot_cutoff
                p.current_bet -= pot.main_pot_cutoff
                pot.main_pot_contributors.add(p)
            if p.current_bet >= pot.side_pot_cutoff:
                pot.side_pot += pot.side_pot_cutoff
                p.current_bet -= pot.side_pot_cutoff
                pot.side_pot_contributors.add(p)
            
            # Return any unallocated chips to the player if they overbet
            if p.current_bet > 0:
                p.chips += p.current_bet
        p.current_bet = 0  # Reset bet after contribution
        
        # Reset player actions after processing
        if not p.folded and not p.is_all_in:
            p.actions_taken = 0

def main(agent, expert_agent, game_idx, num_players):
    # Game parameters
    # num_players = 3
    
    # PPO parameters
    # num_inputs = 19
    # num_outputs = 6
    # lr = 1e-4
    print(game_idx, num_players)
    agent_ids = np.array(range(num_players)) + (game_idx * num_players)
    print(agent_ids)
    # hidden_size = 1024

    # Training parameters
    init_episode = 0
    max_episodes = 1_000_000_000_000
    weight_save_freq = 10_000
    batch_size = 32 # 512
    buffer_size = 256 # batch_size * 20
    episodes = {agent_id: init_episode for agent_id in agent_ids}
    cum_episode = init_episode
    today = date.today().strftime('%Y-%m-%d')

    writer = SummaryWriter(log_dir=os.path.join('train_logs', today))
    weights_folder = os.path.join('weights', today)
    os.makedirs(weights_folder, exist_ok=True)

    # agent = PPO(num_inputs, num_outputs, lr, agent_ids, hidden_size, mini_batch_size=batch_size)
    # agent.load('e500000_2025-02-16.pth', os.path.join('weights', '2025-02-16'))

    while cum_episode < max_episodes:
        starting_chips = 150
        players = [Player(starting_chips, agent_ids[i]) for i in range(num_players)]
        players[0].designated_agent = True
        dealer_position = 0
        num_blind_posted = 1
        num_blind_increase = 0
        small_blind = 5

        cum_reward = 0

        while len(players) > 1:
            deck = Deck()
            is_heads_up = len(players) == 2
            if is_heads_up:
                sb_position = dealer_position
                bb_position = (dealer_position + 1) % 2
            else:
                sb_position = (dealer_position + 1) % len(players)
                bb_position = (dealer_position + 2) % len(players)

            # print('Posting blinds')
            if num_blind_posted % 5 == 0:
                small_blind *= 2
                num_blind_increase += 1
                # print('Small blind increased.')
            players[sb_position].bet(small_blind)  # Small blind
            players[bb_position].bet(2 * small_blind)  # Big blind
            num_blind_posted += 1

            players[dealer_position].position = 0
            players[sb_position].position = 1
            players[bb_position].position = 2

            # Dealing initial hands
            for player in players:
                player.hand = deck.deal(2)
            
            # print('Preflop')
            pot = Pot()
            community_cards = []
            betting_round(agent, players, community_cards, pot, big_blind=small_blind*2, num_blind_increase=num_blind_increase, street=0, current_highest_bet=2*small_blind, starting_player=dealer_position)

            # only have to check this bcs otherwise it doesn't matter
            first_act_pos = bb_position if players[sb_position].is_all_in or players[sb_position].folded else sb_position

            # print('Flop')
            community_cards += deck.deal(3)
            # print(community_cards)
            betting_round(agent, players, community_cards, pot, big_blind=small_blind*2, num_blind_increase=num_blind_increase, street=1, current_highest_bet=0, starting_player=first_act_pos)

            first_act_pos = bb_position if players[sb_position].is_all_in or players[sb_position].folded else sb_position

            # print('Turn')
            community_cards += deck.deal(1)
            # print(community_cards)
            betting_round(agent, players, community_cards, pot, big_blind=small_blind*2, num_blind_increase=num_blind_increase, street=2, current_highest_bet=0, starting_player=first_act_pos)

            first_act_pos = bb_position if players[sb_position].is_all_in or players[sb_position].folded else sb_position

            # print('River')
            community_cards += deck.deal(1)
            # print(community_cards)
            betting_round(agent, players, community_cards, pot, big_blind=small_blind*2, num_blind_increase=num_blind_increase, street=3, current_highest_bet=0, starting_player=first_act_pos)

            # print(f'{pot.main_pot}, {pot.main_pot_contributors = }, {pot.side_pot = }, {pot.side_pot_contributors = }')

            main_pot_contenders = list(pot.main_pot_contributors)
            main_pot_winners = [main_pot_contenders[0]]
            best_eval = evaluate_cards(*[str(e) for e in main_pot_contenders[0].hand + community_cards])
            for i in range(1, len(main_pot_contenders)):
                hand_eval = evaluate_cards(*[str(e) for e in main_pot_contenders[i].hand + community_cards])
                if hand_eval < best_eval:
                    main_pot_winners = [main_pot_contenders[i]]
                    best_eval = hand_eval
                elif hand_eval == best_eval:
                    main_pot_winners.append(main_pot_contenders[i])
            for p in main_pot_winners:
                p.chips += pot.main_pot // len(main_pot_winners)

            if pot.side_pot_contributors:
                side_pot_contenders = list(pot.side_pot_contributors)
                side_pot_winners = [side_pot_contenders[0]]
                best_eval = evaluate_cards(*[str(e) for e in side_pot_contenders[0].hand + community_cards])
                for i in range(1, len(side_pot_contenders)):
                    hand_eval = evaluate_cards(*[str(e) for e in side_pot_contenders[i].hand + community_cards])
                    if hand_eval < best_eval:
                        side_pot_winners = [side_pot_contenders[i]]
                        best_eval = hand_eval
                    elif hand_eval == best_eval:
                        side_pot_winners.append(side_pot_contenders[i])
                for p in side_pot_winners:
                    p.chips += pot.side_pot // len(side_pot_winners)

            # print(f"Final pot size: {pot.main_pot}")
            for p in players:
                reward = p.chips - p.chips_at_start_of_ep
                if p.num_actions_this_episode > 0 and p.designated_agent:
                    cum_reward += reward
                    agent.update_reward_done(p.id, p.num_actions_this_episode, reward)
                p.folded = False
                p.is_all_in = False
                p.actions_taken = 0
                p.current_bet = 0
                p.last_action = -1
                p.num_actions_this_episode = 0
                p.chips_at_start_of_ep = p.chips
                # print(p.hand, p.chips)
                
                episodes[p.id] += 1
                cum_episode += 1

                if agent.num_of_stored_obs(p.id) >= buffer_size:
                    # print('Updating weights...')
                    actor_loss, critic_loss = agent.update(p.id)
                    writer.add_scalar(f'actor_loss{game_idx}', actor_loss, cum_episode)
                    writer.add_scalar(f'critic_loss{game_idx}', critic_loss, cum_episode)
                    writer.flush()

                if cum_episode % weight_save_freq == 0:
                    print(f'avg reward in last {weight_save_freq}: {cum_reward/weight_save_freq}')
                    cum_reward = 0
                    agent.inference = True
                    winrate = test_agent(agent, agent_ids, 1000)
                    agent.inference = False
                    print(f'Winrate in a 1000 games compared to random actions: ({cum_episode - weight_save_freq}-{cum_episode}: {winrate})')
                    print('Saving weights and rewards...')
                    agent.save_weights(f'e{cum_episode}_{today}_proc{game_idx}', weights_folder)
                    writer.add_scalar(f'Winrate_proc{game_idx}', winrate, cum_episode)
                    writer.flush()

            
            players = [p for p in players if p.chips != 0]
            dealer_position = (dealer_position + 1) % len(players)

    writer.close()


num_inputs = 19
num_outputs = 5
lr = 3e-4
hidden_size = 256
num_players = 3
num_parallel_games = 10

def run_game_process(rank, model, device):
    torch.cuda.set_device(device)
    
    agent = PPO(num_inputs, num_outputs, lr, agent_ids=list(range(num_players * num_parallel_games)), hidden_size=hidden_size, inference=False)
    agent.model = model

    expert_agent = PPO(num_inputs, num_outputs, lr, agent_ids=[0], hidden_size=hidden_size, inference=True)
    expert_agent.model = model

    main(agent, expert_agent, rank, num_players)
    
    print(f"Process {rank} finished.")


if __name__ == "__main__":
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Create a shared PPO model
    shared_model = PPO(num_inputs, num_outputs, lr, agent_ids=list(range(num_players * num_parallel_games)), hidden_size=hidden_size, inference=False)
    shared_model.load_pretrained_imitation('pretrained.pth', 'imitation_learning')
    shared_model = shared_model.model
    shared_model.share_memory()  # Make it accessible across processes

    processes = []
    for rank in range(num_parallel_games):
        p = mp.Process(target=run_game_process, args=(rank, shared_model, device))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All games finished.")