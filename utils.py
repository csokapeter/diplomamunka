import numpy as np
from collections import Counter

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


class Card:
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value

    def __repr__(self):
        return f'{self.value}{self.suit}'
