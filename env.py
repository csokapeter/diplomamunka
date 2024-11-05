import itertools
import random
from phevaluator.evaluator import evaluate_cards


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
    def __init__(self, chips):
        self.chips = chips
        self.current_bet = 0
        self.actions_taken = 0
        self.hand = []
        self.folded = False
        self.is_all_in = False

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


def get_player_action(valid_actions):
    action = None
    while action not in valid_actions:
        action = input(f'choose {", ".join(valid_actions)}\n')
    return action


def betting_round(players, pot, current_highest_bet, starting_player=0):
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
            action = None

            # Determine if all other players are all-in or folded
            all_other_players_all_in_or_folded = len(active_players) == 1

            # Check if the current player has to go all-in to call
            must_go_all_in = p.chips <= (current_highest_bet - p.current_bet)

            # If player has to call all-in, or all others are all-in/folded, restrict options to "call" or "fold"
            if must_go_all_in or all_other_players_all_in_or_folded:
                action = get_player_action(['call', 'fold'])
                if action == "fold":
                    p.fold()
                    if p in pot.main_pot_contributors:
                        pot.main_pot_contributors.remove(p)
                    if p in pot.side_pot_contributors:
                        pot.side_pot_contributors.remove(p)
                elif action == "call":
                    p.call(pot.main_pot_cutoff + pot.side_pot_cutoff)
                    if is_sidepot and p.current_bet < pot.side_pot_cutoff:
                        pot.side_pot_cutoff = p.current_bet
                    elif p.current_bet < pot.main_pot_cutoff:
                        pot.side_pot_cutoff = pot.main_pot_cutoff - p.current_bet
                        pot.main_pot_cutoff = p.current_bet
                        
            else:
                if p.current_bet < current_highest_bet:
                    action = get_player_action(['call', 'raise', 'fold'])
                    if action == "fold":
                        p.fold()
                        if p in pot.main_pot_contributors:
                            pot.main_pot_contributors.remove(p)
                        if p in pot.side_pot_contributors:
                            pot.side_pot_contributors.remove(p)
                    elif action == "call":
                        p.call(pot.main_pot_cutoff + pot.side_pot_cutoff)

                    elif action == "raise":
                        raise_amount = int(input('raise amount\n'))
                        actual_raise_amount = p.raise_bet(raise_amount, current_highest_bet)
                        print(f'{actual_raise_amount=}')
                        current_highest_bet += actual_raise_amount
                        print(f'{current_highest_bet=}')
                        print(f'{is_sidepot=}')
                        if is_sidepot:
                            pot.side_pot_cutoff += actual_raise_amount
                        else:
                            pot.main_pot_cutoff += actual_raise_amount

                else:
                    action = get_player_action(['check', 'raise'])
                    if action == "raise":
                        raise_amount = int(input('raise amount\n'))
                        actual_raise_amount = p.raise_bet(raise_amount, current_highest_bet)
                        current_highest_bet += raise_amount
                        print(f'{is_sidepot=}')
                        if is_sidepot:
                            pot.side_pot_cutoff += actual_raise_amount
                        else:
                            pot.main_pot_cutoff += actual_raise_amount

            p.actions_taken += 1

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


#TODO: Currently raising is on top of small blind and big blind and very unintuitive:
# When sb raised 100 and you want to raise it to 110 and you are the SB you have to type 105 in
# Or when you are BB you have to type 100 in...
# Just make it so that you have type in the total amount you want to bet
#TODO: What if you don't have enough for blinds - Good maybe? - final pot size isn't correct when there is a player that is already all in
def main_game():
    num_players = 3
    starting_chips = 150
    players = [Player(starting_chips) for _ in range(num_players)]
    dealer_position = 0

    while len(players) > 1:
        is_heads_up = len(players) == 2
        if is_heads_up:
            sb_position = dealer_position
            bb_position = (dealer_position + 1) % 2
        else:
            sb_position = (dealer_position + 1) % len(players)
            bb_position = (dealer_position + 2) % len(players)

        deck = Deck()

        print('Posting blinds')
        small_blind = 5
        players[sb_position].bet(small_blind)  # Small blind
        players[bb_position].bet(2 * small_blind)  # Big blind

        # Dealing initial hands
        for player in players:
            player.hand = deck.deal(2)
        
        print('Preflop')
        pot = Pot()
        betting_round(players, pot, 2 * small_blind, starting_player=dealer_position)

        # only have to check this bcs otherwise it doesn't matter
        first_act_pos = bb_position if players[sb_position].is_all_in or players[sb_position].folded else sb_position

        print('Flop')
        community_cards = deck.deal(3)
        print(community_cards)
        betting_round(players, pot, 0, starting_player=first_act_pos)

        first_act_pos = bb_position if players[sb_position].is_all_in or players[sb_position].folded else sb_position

        print('Turn')
        community_cards.append(*deck.deal(1))
        print(community_cards)
        betting_round(players, pot, 0, starting_player=first_act_pos)

        first_act_pos = bb_position if players[sb_position].is_all_in or players[sb_position].folded else sb_position

        print('River')
        community_cards.append(*deck.deal(1))
        print(community_cards)
        betting_round(players, pot, 0, starting_player=first_act_pos)

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
        
        players = [p for p in players if p.chips != 0]
        dealer_position = (dealer_position + 1) % len(players)

        print(f"Final pot size: {pot.main_pot}")
        for p in players:
            p.folded = False
            p.is_all_in = False
            p.actions_taken = 0
            p.current_bet = 0
            print(p.hand, p.chips)


if __name__ == '__main__':
    main_game()