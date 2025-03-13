from phevaluator.evaluator import evaluate_cards


class Player:
    def __init__(self, chips, id, takes_random_actions=False):
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
        self.takes_random_actions = takes_random_actions
        self.designated_agent = False

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
        # return f'{self.hand} {self.chips}'
        return f'''
        {self.id=}
        {self.chips=}
        {self.current_bet=}
        {self.actions_taken=}
        {self.hand=}
        {self.folded=}
        {self.is_all_in=}
        {self.position=}
        {self.last_action=}
        {self.num_actions_this_episode=}
        {self.chips_at_start_of_ep=}
        {self.takes_random_actions=}
        '''
