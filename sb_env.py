import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from phevaluator.evaluator import evaluate_cards

from utils import *
from card import Card, Deck
from pot import Pot
from player import Player
from expert_agent import MLP


class PokerGame(gym.Env):
    def __init__(self, random_agents, inference):
        super(PokerGame, self).__init__()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(19,), dtype=np.float32)
        self.starting_chips = 150
        self.num_players = 3
        self.round = 0
        self.valid_action_mask = np.array([])
        self.random_agents = random_agents
        self.is_next_action = True
        self.inference = inference
        self.wins = 0
        self.episodes = 0

        self.expert_agent = MLP(input_size=19, num_classes=5, act='tanh')
        self.expert_agent.load_state_dict(torch.load('imitation_learning/pretrained_tanh.pth'))
        self.expert_agent.eval()

        self.info = dict()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.game(action)
        return obs, reward, terminated, truncated, info 

    def reset(self, seed=None, options=None):
        self.valid_action_mask = np.array([])
        obs, _, _, _, info = self.game(-1)
        return obs, info

    def get_action_mask(self):
        return self.valid_action_mask

    def select_action(self, rnd, obs):
        if rnd:
            valid_indices = np.where(self.valid_action_mask)[0]
            action = np.random.choice(valid_indices)
        else:
            action = self.expert_agent.select_action(obs, self.valid_action_mask)
        return action

    def game(self, action):
        next_obs = None
        reward = 0
        terminated = False
        truncated = False
        if self.round == 0:
            self.players = [Player(self.starting_chips, i) for i in range(self.num_players)]
            self.players[0].designated_agent = True
            self.players[1].takes_random_actions = self.random_agents
            self.players[2].takes_random_actions = self.random_agents
            self.dealer_position = 0
            self.num_blind_posted = 1
            self.num_blind_increase = 0
            self.small_blind = 5
            self.round += 1
            self.first_time_street = True

        if self.round == 1:
            self.deck = Deck()

            is_heads_up = len(self.players) == 2
            if is_heads_up:
                self.sb_position = self.dealer_position
                self.bb_position = (self.dealer_position + 1) % 2
            else:
                self.sb_position = (self.dealer_position + 1) % len(self.players)
                self.bb_position = (self.dealer_position + 2) % len(self.players)

            if self.num_blind_posted % 5 == 0:
                self.small_blind *= 2
                self.num_blind_increase += 1

            self.players[self.sb_position].bet(self.small_blind)  # Small blind
            self.players[self.bb_position].bet(2 * self.small_blind)  # Big blind
            self.num_blind_posted += 1

            self.players[self.sb_position].position = 1
            self.players[self.dealer_position].position = 0
            self.players[self.bb_position].position = 2

            # Dealing initial hands
            for player in self.players:
                player.hand = self.deck.deal(2)
        
            self.pot = Pot()
            self.community_cards = []

            self.current_player = self.dealer_position
            self.round += 1

        if self.round == 2:
            if self.first_time_street:
                self.current_highest_bet = 2 * self.small_blind
            self.first_time_street = False
            next_obs = self.betting_round(action, street=0, starting_player=self.current_player)

        if self.round == 3:
            # only have to check this bcs otherwise it doesn't matter
            if self.first_time_street:
                self.current_highest_bet = 0
                self.current_player = self.bb_position if self.players[self.sb_position].is_all_in or self.players[self.sb_position].folded else self.sb_position
                self.community_cards += self.deck.deal(3)

            self.first_time_street = False
            next_obs = self.betting_round(action, street=1, starting_player=self.current_player)

        if self.round == 4:
            if self.first_time_street:
                self.current_highest_bet = 0
                self.current_player = self.bb_position if self.players[self.sb_position].is_all_in or self.players[self.sb_position].folded else self.sb_position
                self.community_cards += self.deck.deal(1)

            self.first_time_street = False
            next_obs = self.betting_round(action, street=2, starting_player=self.current_player)

        if self.round == 5:
            if self.first_time_street:
                self.current_highest_bet = 0
                self.current_player = self.bb_position if self.players[self.sb_position].is_all_in or self.players[self.sb_position].folded else self.sb_position
                self.community_cards += self.deck.deal(1)

            self.first_time_street = False
            next_obs = self.betting_round(action, street=3, starting_player=self.current_player)

        if self.round == 6:
            des_p = next((player for player in self.players if player.designated_agent), None)
            next_obs = get_obs(des_p.hand, self.community_cards, self.num_blind_increase, street=3, player=des_p, players=self.players)
            self.round = 0
            main_pot_contenders = list(self.pot.main_pot_contributors)
            main_pot_winners = [main_pot_contenders[0]]
            best_eval = evaluate_cards(*[str(e) for e in main_pot_contenders[0].hand + self.community_cards])
            for i in range(1, len(main_pot_contenders)):
                hand_eval = evaluate_cards(*[str(e) for e in main_pot_contenders[i].hand + self.community_cards])
                if hand_eval < best_eval:
                    main_pot_winners = [main_pot_contenders[i]]
                    best_eval = hand_eval
                elif hand_eval == best_eval:
                    main_pot_winners.append(main_pot_contenders[i])
            for p in main_pot_winners:
                p.chips += self.pot.main_pot // len(main_pot_winners)

            if self.pot.side_pot_contributors:
                side_pot_contenders = list(self.pot.side_pot_contributors)
                side_pot_winners = [side_pot_contenders[0]]
                best_eval = evaluate_cards(*[str(e) for e in side_pot_contenders[0].hand + self.community_cards])
                for i in range(1, len(side_pot_contenders)):
                    hand_eval = evaluate_cards(*[str(e) for e in side_pot_contenders[i].hand + self.community_cards])
                    if hand_eval < best_eval:
                        side_pot_winners = [side_pot_contenders[i]]
                        best_eval = hand_eval
                    elif hand_eval == best_eval:
                        side_pot_winners.append(side_pot_contenders[i])
                for p in side_pot_winners:
                    p.chips += self.pot.side_pot // len(side_pot_winners)

            for p in self.players:
                if p == des_p:
                    reward = p.chips - p.chips_at_start_of_ep
                p.folded = False
                p.is_all_in = False
                p.actions_taken = 0
                p.current_bet = 0
                p.last_action = -1
                p.num_actions_this_episode = 0
                p.chips_at_start_of_ep = p.chips

            self.players = [p for p in self.players if p.chips != 0]
            self.dealer_position = (self.dealer_position + 1) % len(self.players)
            if des_p.chips == 0 or len(self.players) == 1:
                terminated = True
                self.round = 0
                if self.inference:
                    self.episodes += 1
                    if len(self.players) == 1 and self.players[0].designated_agent:
                        self.wins += 1
                        print(f'{self.episodes=}, {self.wins=}')
            else:
                self.round = 1
            
            if not self.inference:
                terminated = True
        
        if action == -1 and terminated:
            self.round = 0
            self.reset()

        if self.inference:
            for p in self.players:
                self.info[f'player{p.id}_hand'] = [card.get_numeric_value() for card in p.hand] + [0] * max(0, 2 - len(p.hand))
                self.info[f'player{p.id}_last_action'] = p.last_action
                self.info[f'player{p.id}_chips'] = p.chips
                self.info[f'player{p.id}_position'] = p.position
                self.info[f'player{p.id}_folded'] = int(p.folded)
            self.info['community_cards'] = [card.get_numeric_value() for card in self.community_cards] + [0] * max(0, 5 - len(self.community_cards))
            self.info['main_pot'] = self.pot.main_pot
            self.info['side_pot'] = self.pot.side_pot

        return next_obs, reward, terminated, truncated, self.info

    def betting_round(self, action, street, starting_player=0):
        num_players = len(self.players)
        self.current_player = starting_player
        self.pot.main_pot_cutoff = self.current_highest_bet
        self.pot.side_pot_cutoff = 0
        big_blind = 2 * self.small_blind

        for p in self.players:
            if p.is_all_in:
                p.actions_taken = 1
                self.pot.main_pot_cutoff = p.current_bet
                self.pot.side_pot_cutoff = self.current_highest_bet - self.pot.main_pot_cutoff
                is_sidepot = True

        while not all((p.current_bet == self.current_highest_bet or p.is_all_in or p.folded) and p.actions_taken > 0 for p in self.players):
            p = self.players[self.current_player]
            all_in_players = [player for player in self.players if player.is_all_in]
            is_sidepot = len(all_in_players) > 0

            active_players = [player for player in self.players if not player.folded and not player.is_all_in]
            if len(active_players) == 1 and p.current_bet == self.current_highest_bet:
                p.actions_taken += 1
                self.current_player = (self.current_player + 1) % num_players
                continue

            if not p.folded and not p.is_all_in:
                obs = get_obs(p.hand, self.community_cards, self.num_blind_increase, street, p, self.players)

                # Determine if all other players are all-in or folded
                all_other_players_all_in_or_folded = len(active_players) == 1

                # Check if the current player has to go all-in to call
                must_go_all_in = p.chips <= (self.current_highest_bet - p.current_bet)

                # If player has to call all-in, or all others are all-in/folded, restrict options to "call" or "fold"
                if must_go_all_in or all_other_players_all_in_or_folded:
                    # action = get_player_action(['fold', 'call'])
                    self.valid_action_mask = np.array([1, 0, 1, 0, 0])
                    if p.designated_agent:
                        if self.is_next_action:
                            self.is_next_action = False
                            return obs
                        else:
                            action = action
                            self.is_next_action = True
                    else:
                        action = self.select_action(p.takes_random_actions, obs)
                    p.last_action = action
                    if action == 0:
                        p.fold()
                        if p in self.pot.main_pot_contributors:
                            self.pot.main_pot_contributors.remove(p)
                        if p in self.pot.side_pot_contributors:
                            self.pot.side_pot_contributors.remove(p)
                    elif action == 2:
                        p.call(self.current_highest_bet)
                        if is_sidepot and p.current_bet < self.pot.side_pot_cutoff:
                            self.pot.side_pot_cutoff = p.current_bet
                        elif p.current_bet < self.pot.main_pot_cutoff:
                            self.pot.side_pot_cutoff = self.pot.main_pot_cutoff - p.current_bet
                            self.pot.main_pot_cutoff = p.current_bet
                            
                else:
                    if p.current_bet < self.current_highest_bet:
                        # action = get_player_action(['fold', 'call', 'raise'])
                        self.valid_action_mask = np.array([1, 0, 1, 1, 1])
                        if p.designated_agent:
                            if self.is_next_action:
                                self.is_next_action = False
                                return obs
                            else:
                                action = action
                                self.is_next_action = True
                        else:
                            action = self.select_action(p.takes_random_actions, obs)
                        p.last_action = action
                        if action == 0:
                            p.fold()
                            if p in self.pot.main_pot_contributors:
                                self.pot.main_pot_contributors.remove(p)
                            if p in self.pot.side_pot_contributors:
                                self.pot.side_pot_contributors.remove(p)
                        elif action == 2:
                            p.call(self.pot.main_pot_cutoff + self.pot.side_pot_cutoff)
                        elif action > 2:
                            if action == 3:
                                raise_amount = self.current_highest_bet + 2 * big_blind
                            elif action == 4:
                                raise_amount = self.current_highest_bet + 15 * big_blind
                            actual_raise_amount = p.raise_bet(raise_amount, self.current_highest_bet)
                            self.current_highest_bet += actual_raise_amount
                            if is_sidepot:
                                self.pot.side_pot_cutoff += actual_raise_amount
                            else:
                                self.pot.main_pot_cutoff += actual_raise_amount

                    else:
                        # action = get_player_action(['check', 'raise'])
                        self.valid_action_mask = np.array([0, 1, 0, 1, 1])
                        if p.designated_agent:
                            if self.is_next_action:
                                self.is_next_action = False
                                return obs
                            else:
                                action = action
                                self.is_next_action = True
                        else:
                            action = self.select_action(p.takes_random_actions, obs)
                        p.last_action = action
                        if action > 2:
                            if action == 3:
                                raise_amount = self.current_highest_bet + 2 * big_blind
                            elif action == 4:
                                raise_amount = self.current_highest_bet + 15 * big_blind
                            actual_raise_amount = p.raise_bet(raise_amount, self.current_highest_bet)
                            self.current_highest_bet += actual_raise_amount
                            if is_sidepot:
                                self.pot.side_pot_cutoff += actual_raise_amount
                            else:
                                self.pot.main_pot_cutoff += actual_raise_amount

                p.actions_taken += 1
                p.num_actions_this_episode += 1

            self.current_player = (self.current_player + 1) % num_players

        all_in_players = [player for player in self.players if player.is_all_in]
        is_sidepot = len(all_in_players) > 0
        self.round += 1
        self.first_time_street = True
        
        # Iterate over all players and properly allocate their contributions
        for p in self.players:
            if p.folded:
                # A folded player's bet should go to the main pot or side pot
                if not is_sidepot:  # No side pot, everything goes to the main pot
                    self.pot.main_pot += p.current_bet
                else:  # Side pot exists, so allocate to side pot if necessary
                    if p.current_bet <= self.pot.main_pot_cutoff:
                        self.pot.main_pot += p.current_bet
                    else:
                        # If they contributed more than the main pot cutoff, allocate excess to side pot
                        self.pot.main_pot += self.pot.main_pot_cutoff
                        self.pot.side_pot += p.current_bet - self.pot.main_pot_cutoff

            elif p.is_all_in:
                # All-in player's bet should be split between the main and side pots
                if p.current_bet <= self.pot.main_pot_cutoff:
                    self.pot.main_pot += p.current_bet
                    self.pot.main_pot_contributors.add(p)
                else:
                    self.pot.main_pot += self.pot.main_pot_cutoff
                    self.pot.side_pot += p.current_bet - self.pot.main_pot_cutoff
                    self.pot.main_pot_contributors.add(p)
                    self.pot.side_pot_contributors.add(p)

            else:  # Active player who hasn't folded or gone all-in
                # Allocate to the main pot or side pot based on their current bet
                if p.current_bet >= self.pot.main_pot_cutoff:
                    self.pot.main_pot += self.pot.main_pot_cutoff
                    p.current_bet -= self.pot.main_pot_cutoff
                    self.pot.main_pot_contributors.add(p)
                if p.current_bet >= self.pot.side_pot_cutoff:
                    self.pot.side_pot += self.pot.side_pot_cutoff
                    p.current_bet -= self.pot.side_pot_cutoff
                    self.pot.side_pot_contributors.add(p)
                
                # Return any unallocated chips to the player if they overbet
                if p.current_bet > 0:
                    p.chips += p.current_bet
            p.current_bet = 0  # Reset bet after contribution
            
            # Reset player actions after processing
            if not p.folded and not p.is_all_in:
                p.actions_taken = 0
