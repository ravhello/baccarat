class Player:
    """
    A player in the team.
    """
    def __init__(self, name, role, counting_values):
        self.name = name
        self.role = role
        self.bankroll = 0
        self.ideal_bankroll = 0
        self.bet_choice = None
        self.bet_amount = 0
        self.sidebets_chosen_and_relative_amounts = {}  # Store side bet decisions and amounts
        self.running_count_per_side = {}  # Running count for each side bet
        self.sidebets_assigned = []  # List of assigned side bets
        self.exchanges_used = 0  # Number of exchanges used
        self.total_bet_amount = 0  # Total bet amount
        self.num_assigned_sides = 0  # Number of assigned side bets
        self.counting_values = counting_values
        self.total_win_for_each_bet = {side: 0 for side in ['Banker', 'Player', 'Tie'] + list(counting_values)}  # Total win for each bet

    def update_running_count(self, card):
        """
        Update the running count for each side bet based on the card drawn.
        """
        for side in self.sidebets_assigned:
            # print(f"{self.name} sees the card {card} and for the side {side} update his running count from {self.running_count_per_side[side]} to {self.running_count_per_side[side] + counting_values[side][card]}")
            self.running_count_per_side[side] += self.counting_values[side][card]

    def reset_session(self):
        """
        Reset the player's session by resetting running counts, exchanges used, and bet information.
        """
        for side in self.running_count_per_side:
            self.running_count_per_side[side] = 0
        self.exchanges_used = 0
        self.reset_bet()

    def reset_bet(self):
        """
        Reset the player's bet by clearing bet choices, amounts, and related information.
        """
        self.bet_choice = None
        self.bet_amount = 0
        self.sidebets_chosen_and_relative_amounts = {}
        self.total_bet_amount = 0
        self.ideal_bankroll = 0
        self.total_win_for_each_bet = {side: 0 for side in ['Banker', 'Player', 'Tie'] + list(self.counting_values)}
