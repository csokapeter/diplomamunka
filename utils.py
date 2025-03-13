import numpy as np
from collections import Counter
from phevaluator.evaluator import evaluate_cards

from poker_hands import poker_hand_lookup


def get_num_gutshots(hand, community_cards):
    value_order = np.array(list("A23456789TJQKA"))
    all_cards = hand + community_cards
    unique_values = sorted(set(card.value for card in all_cards), key=lambda x: np.where(value_order == x)[0][0])

    mask = np.isin(value_order, unique_values).astype(int)
    pattern1 = np.array([1, 1, 0, 1, 1])
    pattern2 = np.array([1, 0, 1, 1, 1])
    pattern3 = np.array([1, 1, 1, 0, 1])
    match_counts = 0
    match_counts += np.sum(np.all(np.lib.stride_tricks.sliding_window_view(mask, len(pattern1)) == pattern1, axis=1))
    match_counts += np.sum(np.all(np.lib.stride_tricks.sliding_window_view(mask, len(pattern2)) == pattern2, axis=1))
    match_counts += np.sum(np.all(np.lib.stride_tricks.sliding_window_view(mask, len(pattern3)) == pattern3, axis=1))

    return match_counts


def get_num_gutshots_community(community_cards):
    value_order = np.array(list("A23456789TJQKA"))
    unique_values = sorted(set(card.value for card in community_cards), key=lambda x: np.where(value_order == x)[0][0])

    mask = np.isin(value_order, unique_values).astype(int)
    pattern1 = np.array([1, 1, 0, 1, 1])
    pattern2 = np.array([1, 0, 1, 1, 1])
    pattern3 = np.array([1, 1, 1, 0, 1])
    match_counts = 0
    match_counts += np.sum(np.all(np.lib.stride_tricks.sliding_window_view(mask, len(pattern1)) == pattern1, axis=1))
    match_counts += np.sum(np.all(np.lib.stride_tricks.sliding_window_view(mask, len(pattern2)) == pattern2, axis=1))
    match_counts += np.sum(np.all(np.lib.stride_tricks.sliding_window_view(mask, len(pattern3)) == pattern3, axis=1))

    return match_counts


def has_open_ended_draw(hand, community_cards):
    value_order = np.array(list("A23456789TJQKA"))
    all_cards = hand + community_cards
    unique_values = sorted(set(card.value for card in all_cards), key=lambda x: np.where(value_order == x)[0][0])

    mask = np.isin(value_order, unique_values).astype(int)
    pattern = np.array([0, 1, 1, 1, 1, 0])
    match_counts = np.sum(np.all(np.lib.stride_tricks.sliding_window_view(mask, len(pattern)) == pattern, axis=1))

    return match_counts


def get_no_suit_hero(hand, community_cards):
    all_cards = hand + community_cards
    suit_counts = Counter(card.suit for card in all_cards)

    return max(suit_counts[hand[0].suit], suit_counts[hand[1].suit])


def get_no_suit_villain(hand, community_cards):
    all_cards = hand + community_cards
    suit_counts = Counter(card.suit for card in all_cards)
    player_suits = {hand[0].suit, hand[1].suit}
    villain_suit_counts = {suit: count for suit, count in suit_counts.items() if suit not in player_suits}

    return max(villain_suit_counts.values(), default=0)


def get_no_same_cards(hand, community_cards):
    all_cards = hand + community_cards
    value_counts = Counter(card.value for card in all_cards)

    return max(value_counts[hand[0].value], value_counts[hand[1].value])


def has_two_pairs(hand, community_cards):
    all_cards = hand + community_cards
    value_counts = Counter(card.value for card in all_cards)
    c1 = value_counts[hand[0].value]
    c2 = value_counts[hand[1].value]

    return int(c1 == 2 and c2 == 2 and len(value_counts) > 1)


def has_straight(hand, community_cards):
    value_order = np.array(list("A23456789TJQKA"))
    all_cards = hand + community_cards
    unique_values = sorted(set(card.value for card in all_cards), key=lambda x: np.where(value_order == x)[0][0])

    mask = np.isin(value_order, unique_values).astype(int)
    pattern = np.array([1, 1, 1, 1, 1])
    match_counts = np.sum(np.all(np.lib.stride_tricks.sliding_window_view(mask, len(pattern)) == pattern, axis=1))

    return int(match_counts >= 1)


def has_full_house(hand, community_cards):
    all_cards = hand + community_cards
    value_counts = Counter(card.value for card in all_cards).values()
    
    return int(3 in value_counts and 2 in value_counts)


def has_straight_flush(hand, community_cards):
    value_order = np.array(list("A23456789TJQKA"))
    all_cards = hand + community_cards
    unique_values = sorted(set(card.value for card in all_cards), key=lambda x: np.where(value_order == x)[0][0])

    mask = np.isin(value_order, unique_values).astype(int)
    pattern = np.array([1, 1, 1, 1, 1])
    match_indices = np.where(np.all(np.lib.stride_tricks.sliding_window_view(mask, len(pattern)) == pattern, axis=1))[0]

    for idx in match_indices:
        straight_values = value_order[idx:idx + 5]
        straight_cards = [card for card in all_cards if card.value in straight_values]

        suit_groups = {}
        for card in straight_cards:
            suit_groups.setdefault(card.suit, []).append(card)

        if any(len(cards) >= 5 for cards in suit_groups.values()):
            return 1

    return 0


def get_card_index(hand):
    ranks = "AKQJT98765432"
    
    card1, card2 = hand
    value1, suit1 = card1.value, card1.suit
    value2, suit2 = card2.value, card2.suit

    if ranks.index(value1) < ranks.index(value2):
        high, low = value1, value2
    else:
        high, low = value2, value1

    suited = suit1 == suit2
    
    if suited:
        return poker_hand_lookup.get(f'{high}{low}s')
    else:
        return poker_hand_lookup.get(f'{high}{low}o')


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
    card_idx = get_card_index(hand) / 100
    num_gutshots = get_num_gutshots(hand, community_cards) / 2
    num_gutshots_community = get_num_gutshots_community(community_cards) / 2
    open_ended = has_open_ended_draw(hand, community_cards)
    hero_no_s = get_no_suit_hero(hand, community_cards) / 7
    villain_no_s = get_no_suit_villain(hand, community_cards) / 5
    no_same_cards = get_no_same_cards(hand, community_cards) / 4
    two_pairs = has_two_pairs(hand, community_cards)
    straight = has_straight(hand, community_cards)
    full_house = has_full_house(hand, community_cards)
    straight_flush = has_straight_flush(hand, community_cards)
    player_pos = player.position / 3
    
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
        stack_of_players = (len(bins) - np.digitize([players[sb_pos].chips, players[bb_pos].chips, players[dealer_pos].chips], bins) - 1)
        last_actions = [players[i].last_action for i in [sb_pos, bb_pos, dealer_pos] if players[i].position != player.position]
    else:
        stack_of_players = len(bins) - np.digitize([players[sb_pos].chips, players[bb_pos].chips, 0], bins) - 1
        last_actions = [players[i].last_action for i in [sb_pos, bb_pos] if players[i].position != player.position]
        last_actions.append(-1)

    stack_of_players = np.array(stack_of_players) / 7 # / 450
    last_actions = np.array(last_actions) / 6
    
    
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


# def get_obs(hand, community_cards, num_blind_increase, street, player, players):
#     card_idx = get_card_index(hand) / 100
#     if len(community_cards) == 0:
#         hand_eval = -1
#     else:
#         hand_eval = evaluate_cards(*[str(h) for h in hand + community_cards]) / 7462
#     player_pos = player.position / 3
    
#     # not sure this is correct
#     pos = 0
#     if len(players) == 2:
#         dealer_pos = 2
#     for p in players:
#         if p.position == 0:
#             dealer_pos = pos
#         elif p.position == 1:
#             sb_pos = pos
#         else:
#             bb_pos = pos
#         pos += 1

#     bins = np.array([0, 37.5, 75, 112.5, 150, 225, 300, 450])
#     if len(players) > 2:
#         stack_of_players = (len(bins) - np.digitize([players[sb_pos].chips, players[bb_pos].chips, players[dealer_pos].chips], bins) - 1)
#         last_actions = [players[i].last_action for i in [sb_pos, bb_pos, dealer_pos] if players[i].position != player.position]
#     else:
#         stack_of_players = len(bins) - np.digitize([players[sb_pos].chips, players[bb_pos].chips, 0], bins) - 1
#         last_actions = [players[i].last_action for i in [sb_pos, bb_pos] if players[i].position != player.position]
#         last_actions.append(-1)

#     stack_of_players = np.array(stack_of_players) / 7 # / 450
#     last_actions = np.array(last_actions) / 6
    
    
#     obs = [card_idx,
#            hand_eval,
#            num_blind_increase,
#            street,
#            player_pos,
#            *stack_of_players,
#            *last_actions]

#     return obs