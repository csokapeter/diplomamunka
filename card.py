import random
import itertools


class Card:
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value

    def get_numeric_value(self):
        suit_map = {'s': 100, 'h': 200, 'd': 300, 'c': 400}
        value_map = {
            'A': 1, '2': 2, '3': 3, '4': 4, '5': 5,
            '6': 6, '7': 7, '8': 8, '9': 9,
            'T': 10, 'J': 11, 'Q': 12, 'K': 13
        }

        return suit_map[self.suit] + value_map[self.value]

    def __repr__(self):
        return f'{self.value}{self.suit}'


class Deck:
    def __init__(self):
        suits = 'shdc'
        values = '2 3 4 5 6 7 8 9 T J Q K A'.split()
        # suits = 'sh'
        # values = '2 3 4 5 6 7 8 9 T J Q K A'.split()
        self.deck = [Card(s, v) for s, v in itertools.product(suits, values)]
        random.shuffle(self.deck)

    def deal(self, num_cards):
        return [self.deck.pop() for _ in range(num_cards)]
    