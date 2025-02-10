import time
import torch
import random
import itertools
import numpy as np
from datetime import date
from phevaluator.evaluator import evaluate_cards

from PPO import PPO
from utils import *


class Card:
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value

    def __repr__(self):
        return f'{self.value}{self.suit}'


class Deck:
    def __init__(self):
        suits = 'shdc'
        values = '2 3 4 5 6 7 8 9 T J Q K A'.split()
        self.deck = [Card(s, v) for s, v in itertools.product(suits, values)]
        random.shuffle(self.deck)

    def deal(self, num_cards):
        return [self.deck.pop() for _ in range(num_cards)]


class Pot():
    def __init__(self):
        self.main_pot = 0
        self.main_pot_cutoff = 0
        self.main_pot_contributors = set()
        self.side_pot = 0
        self.side_pot_cutoff = 0
        self.side_pot_contributors = set()


class Player:
    def __init__(self, chips, id):
        self.id = id
        self.chips = chips
        self.current_bet = 0
        self.actions_taken = 0
        self.hand = []
        self.folded = False
        self.is_all_in = False
        self.position = -1
        self.last_action = -1
        self.num_actions_this_episode = 0
        self.chips_at_start_of_ep = chips

    def bet(self, amount):
        actual_bet_amount = min(self.chips, amount)
        self.chips -= actual_bet_amount
        self.current_bet += actual_bet_amount
        if self.chips == 0:
            self.is_all_in = True
        return actual_bet_amount

    def call(self, current_highest_bet):
        actual_call_amount = min(self.chips, current_highest_bet - self.current_bet)
        self.bet(actual_call_amount)
        return actual_call_amount

    def raise_bet(self, amount, current_highest_bet):
        amount -= self.call(current_highest_bet)
        actual_bet_amount = self.bet(amount)
        return actual_bet_amount

    def fold(self):
        self.folded = True

    def evaluate_hand(self, community_cards):
        self.hand_eval = evaluate_cards(*[str(e) for e in self.hand + community_cards])
        return self.hand_eval

    def __repr__(self):
        return f'{self.hand} {self.chips}'

# Actions:
# 0 - fold
# 1 - check
# 2 - call
# 3 - 2 BB raise
# 4 - 4 BB raise
# 5 - all in

# card combo //1 AA = 1, 72o = 100
# num gutshot //1: 0,1,2
# num gutshots in community cards //1: 0,1,2
# player has open ended draw //1
# number of cards of same suit for player //1 [1-7]
# number of cards of same suit for not player //1 [1-5]
# number of same cards //1 [1-4]
# does player have two pairs //1
# does player have straight //1
# does player have full house //1
# does player have straight flush //1
# How many times the blinds went up //1
# street //1 [preflop, flop, turn, river]
# player position //1 [dealer, sb, bb]
# stack of players //3 [300%-200%, 200-150%, 150%-100%, 100%-75%, 75%-50%, 50%-25%, 25%-0%] of starting stack order: sb, bb, dealer
# other players last actions //2 how much they bet [1-6] - fold, check, min raise, raise 50%, raise 100%, all-in order: sb, bb, dealer

def get_obs(hand, community_cards, num_blind_increase, street, player, players):
    card_idx = get_card_index(hand)
    num_gutshots = get_num_gutshots(hand, community_cards)
    num_gutshots_community = get_num_gutshots_community(community_cards)
    open_ended = has_open_ended_draw(hand, community_cards)
    hero_no_s = get_no_suit_hero(hand, community_cards)
    villain_no_s = get_no_suit_villain(hand, community_cards)
    no_same_cards = get_no_same_cards(hand, community_cards)
    two_pairs = has_two_pairs(hand, community_cards)
    straight = has_straight(hand, community_cards)
    full_house = has_full_house(hand, community_cards)
    straight_flush = has_straight_flush(hand, community_cards)
    player_pos = player.position
    
    # not sure this is correct
    pos = 0
    if len(players) == 2:
        dealer_pos = 2
    for p in players:
        if p.position == 0:
            dealer_pos = pos
        elif p.position == 1:
            sb_pos = pos
        else:
            bb_pos = pos
        pos += 1

    bins = np.array([0, 37.5, 75, 112.5, 150, 225, 300, 450])
    if len(players) > 2:
        stack_of_players = len(bins) - np.digitize([players[sb_pos].chips, players[bb_pos].chips, players[dealer_pos].chips], bins) - 1
        last_actions = [players[i].last_action for i in [sb_pos, bb_pos, dealer_pos] if players[i].position != player.position]
    else:
        stack_of_players = len(bins) - np.digitize([players[sb_pos].chips, players[bb_pos].chips, 0], bins) - 1
        last_actions = [players[i].last_action for i in [sb_pos, bb_pos] if players[i].position != player.position]
        last_actions.append(-1)
    
    
    obs = [card_idx,
           num_gutshots,
           num_gutshots_community,
           open_ended,
           hero_no_s,
           villain_no_s,
           no_same_cards,
           two_pairs,
           straight,
           full_house,
           straight_flush,
           num_blind_increase,
           street,
           player_pos,
           *stack_of_players,
           *last_actions]

    return obs


def get_player_action(valid_actions):
    action = None
    while action not in valid_actions:
        action = input(f'choose {", ".join(valid_actions)}\n')
    return action


def betting_round(agent, players, community_cards, pot, big_blind, num_blind_increase, street, current_highest_bet, starting_player=0):
    num_players = len(players)
    current_player = starting_player
    pot.main_pot_cutoff = current_highest_bet
    pot.side_pot_cutoff = 0

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
            print(p.hand, p.chips)
            obs = get_obs(p.hand, community_cards, num_blind_increase, street, p, players)
            print(obs)

            # Determine if all other players are all-in or folded
            all_other_players_all_in_or_folded = len(active_players) == 1

            # Check if the current player has to go all-in to call
            must_go_all_in = p.chips <= (current_highest_bet - p.current_bet)

            # If player has to call all-in, or all others are all-in/folded, restrict options to "call" or "fold"
            if must_go_all_in or all_other_players_all_in_or_folded:
                # action = get_player_action(['call', 'fold'])
                valid_action_mask = torch.tensor([1., 0., 1., 0., 0., 0.])
                action = agent.select_action(p.id, torch.tensor(obs, dtype=torch.float32), valid_action_mask)
                p.last_action = action
                print(action)
                if action == 0:
                    p.fold()
                    if p in pot.main_pot_contributors:
                        pot.main_pot_contributors.remove(p)
                    if p in pot.side_pot_contributors:
                        pot.side_pot_contributors.remove(p)
                elif action == 2:
                    p.call(pot.main_pot_cutoff + pot.side_pot_cutoff)
                    if is_sidepot and p.current_bet < pot.side_pot_cutoff:
                        pot.side_pot_cutoff = p.current_bet
                    elif p.current_bet < pot.main_pot_cutoff:
                        pot.side_pot_cutoff = pot.main_pot_cutoff - p.current_bet
                        pot.main_pot_cutoff = p.current_bet
                        
            else:
                if p.current_bet < current_highest_bet:
                    # action = get_player_action(['call', 'raise', 'fold'])
                    valid_action_mask = torch.tensor([1., 0., 1., 1., 1., 1.])
                    action = agent.select_action(p.id, torch.tensor(obs, dtype=torch.float32), valid_action_mask)
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
                        elif action == 4:
                            raise_amount = current_highest_bet + 4 * big_blind
                        elif action == 5:
                            raise_amount = current_highest_bet + 15 * big_blind
                        actual_raise_amount = p.raise_bet(raise_amount, current_highest_bet)
                        current_highest_bet += actual_raise_amount
                        if is_sidepot:
                            pot.side_pot_cutoff += actual_raise_amount
                        else:
                            pot.main_pot_cutoff += actual_raise_amount

                else:
                    # action = get_player_action(['check', 'raise'])
                    valid_action_mask = torch.tensor([0., 1., 0., 1., 1., 1.])
                    action = agent.select_action(p.id, torch.tensor(obs, dtype=torch.float32), valid_action_mask)
                    p.last_action = action
                    if action > 2:
                        if action == 3:
                            raise_amount = current_highest_bet + (2 - (current_highest_bet == big_blind)) * big_blind
                        elif action == 4:
                            raise_amount = current_highest_bet + 4 * big_blind
                        elif action == 5:
                            raise_amount = current_highest_bet + 15 * big_blind
                        actual_raise_amount = p.raise_bet(raise_amount, current_highest_bet)
                        current_highest_bet += raise_amount
                        if is_sidepot:
                            pot.side_pot_cutoff += actual_raise_amount
                        else:
                            pot.main_pot_cutoff += actual_raise_amount

            p.actions_taken += 1
            agent.store_obs(p.id)
            print(f'{action =}')
            p.num_actions_this_episode += 1
            # input('q')

        current_player = (current_player + 1) % num_players
        for p in players:
            print(p.hand, p.folded, p.current_bet, p.actions_taken, current_highest_bet)
            print()

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

# Make it parallel somehow
def main():
    tm = time.time()
    # Game parameters
    num_players = 3
    
    # PPO parameters
    num_inputs = 19
    num_outputs = 6
    lr = 1e-4
    agent_ids = list(range(num_players))
    hidden_size = 1024

    # Training parameters
    init_episode = 0
    max_episodes = 1_000_000_000_000
    weight_save_freq = 10 # 100_000
    batch_size = 10 # 1024
    buffer_size = batch_size * 10 # 40
    episodes = {agent_id: init_episode for agent_id in agent_ids}
    cum_episode = init_episode
    rewards = []
    today = date.today().strftime('%Y-%m-%d')

    agent = PPO(num_inputs, num_outputs, lr, agent_ids, hidden_size, mini_batch_size=batch_size)

    while cum_episode < max_episodes:
        starting_chips = 150
        players = [Player(starting_chips, i) for i in range(num_players)]
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

            print('Posting blinds')
            if num_blind_posted % 5 == 0:
                small_blind *= 2
                num_blind_increase += 1
                print('Small blind increased.')
            players[sb_position].bet(small_blind)  # Small blind
            players[bb_position].bet(2 * small_blind)  # Big blind
            num_blind_posted += 1

            players[dealer_position].position = 0
            players[sb_position].position = 1
            players[bb_position].position = 2

            # Dealing initial hands
            for player in players:
                player.hand = deck.deal(2)
            
            print('Preflop')
            pot = Pot()
            community_cards = []
            betting_round(agent, players, community_cards, pot, big_blind=small_blind*2, num_blind_increase=num_blind_increase, street=0, current_highest_bet=2*small_blind, starting_player=dealer_position)

            # only have to check this bcs otherwise it doesn't matter
            first_act_pos = bb_position if players[sb_position].is_all_in or players[sb_position].folded else sb_position

            print('Flop')
            community_cards += deck.deal(3)
            print(community_cards)
            betting_round(agent, players, community_cards, pot, big_blind=small_blind*2, num_blind_increase=num_blind_increase, street=1, current_highest_bet=0, starting_player=first_act_pos)

            first_act_pos = bb_position if players[sb_position].is_all_in or players[sb_position].folded else sb_position

            print('Turn')
            community_cards += deck.deal(1)
            print(community_cards)
            betting_round(agent, players, community_cards, pot, big_blind=small_blind*2, num_blind_increase=num_blind_increase, street=2, current_highest_bet=0, starting_player=first_act_pos)

            first_act_pos = bb_position if players[sb_position].is_all_in or players[sb_position].folded else sb_position

            print('River')
            community_cards += deck.deal(1)
            print(community_cards)
            betting_round(agent, players, community_cards, pot, big_blind=small_blind*2, num_blind_increase=num_blind_increase, street=3, current_highest_bet=0, starting_player=first_act_pos)

            print(f'{pot.main_pot}, {pot.main_pot_contributors = }, {pot.side_pot = }, {pot.side_pot_contributors = }')

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

            print(f"Final pot size: {pot.main_pot}")
            for p in players:
                reward = p.chips - p.chips_at_start_of_ep
                agent.update_reward_done(p.id, p.num_actions_this_episode, reward)
                p.folded = False
                p.is_all_in = False
                p.actions_taken = 0
                p.current_bet = 0
                p.last_action = -1
                p.num_actions_this_episode = 0
                p.chips_at_start_of_ep = p.chips
                print(p.hand, p.chips)
                
                episodes[p.id] += 1
                cum_episode += 1
                rewards.append(reward)

                print(f'{agent.num_of_stored_obs(p.id) =}')
                if agent.num_of_stored_obs(p.id) >= buffer_size:
                    print(episodes, cum_episode)
                    print('Updating weights...')
                    agent.update(p.id)
                    print(tm, time.time())
                    exit()

                # Mean reward will always be zero
                # Test it against random actions
                if cum_episode % weight_save_freq == 0:
                    mean_reward = np.mean(rewards[-weight_save_freq:])
                    print(f'Mean reward between ({cum_episode - weight_save_freq}-{cum_episode}: {mean_reward})')
                    print('Saving weights and rewards')
                    #TODO: save weights and rewards

            
            players = [p for p in players if p.chips != 0]
            dealer_position = (dealer_position + 1) % len(players)


if __name__ == '__main__':
    main()
