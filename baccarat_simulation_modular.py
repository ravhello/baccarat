import math
import random
import os
import pandas as pd
import statistics
from itertools import combinations
import numpy as np
import cProfile
import pstats
import time
import multiprocessing

from player_class import *

# We keep track of the time spent running, so that we know instantly the impact of any modification 
start_time = time.perf_counter()

# Here we measure all the functions performances, so that we know where we are wasting most of our time
# and we can optimize it
profiler = cProfile.Profile()
profiler.enable()

# Variables to optimize, that are passed from the main script. If not, the default variables here will be used.

num_players = 4                     # total number of players of the team
num_punters = 4                     # total number of punters (the others will be considered automatically as counters)
exchanges_all_in_one = False        # if, when one player goes bankrupt, all the players will redistribute their money
                                    # to the initial default percentages or we will proceed by single transfers of cash from one player to another
                                    # This depends on the casino conditions. We want to avoid transfers or the casino will detect the team sooner
bankroll_punters_wrt_counters = 3    # bankroll of the punters in respect to the counters
punter_base_bet_multiplier = 1      # how big is the base bet of a punter in respect to a counter
side_multiplier_wrt_base = 1        # how many times bigger the side bet is compared to the base bet of the punter
kelly_multiplier = 5                # multiplier for the Kelly criterion

# Constants and initial settings (depending on the casino conditions, these settings may vary)
# Some of the variables here are considered as constants despite we could change them to optimize it
# This is because we already optimized them before, and we discovered that the simulations performs better in this way
# and they are not interdependent with the other variables

num_decks = 8                        # number of decks dealt for the game (may vary from 6 to 8)
initial_stake = 10000 * num_players  # euros that every player is investing for the project
max_side_counts_per_player = 2       # number of single-ties that every player will be able to count at the same time
cutoff = 52                          # the average position for the stop card (in general it is positioned one deck to the end)
                                     # so the decks dealt (penetration) are 7
std_cutoff = 3                       # standard deviation of the cutoff card placed by the dealer (3 cards error from 1 deck)
num_trips = 10000
num_sessions = 5
hours_per_session = 7
hands_per_hour = 100
chips_types_list = [0.25, 0.5, 1, 2, 5, 10, 20, 50, 60, 70, 80, 90, 100]   # possible types of chips
bet_min_main = 10                   # main bets and side bets (single-ties) limits
bet_max_main = 2000
bet_min_side = 1
bet_max_side = 25
exchanges_allowed_per_player_per_session = 20   # single transfers (when someone's bankruptcy) allowed per player each session
counters_base_bet_percentages = [0.50, 0.45, 0.05]   # distribution of the bets when no advantage is available, or when we have to bet on the main to be able to bet on the side
punters_base_bet_percentages = [0.50, 0.45, 0.05]    # 50% bet on Player, 45% Banker, 5% Tie
                                                    # This is to minimize the money lost when there is no advantage (Tie bet has the lowest expected return on average)
                                                    # A main bet is always mandatory if we want to bet also on a side bet
average_size_main_bet = 10
max_limit_bet = 100                 # maximum limit we pose to the main bets
base_bet_std_perc = 0.2             # standard deviation of the main bet
punters_playing_sides_perc = 0      # If side size is proportionate to the bankroll, it becomes the percentage of punters BEYOND THE MINIMUM NUMBER TO ACHIEVE THE CORRECT SIZE
sides_perc_punters_bet_on = 0       # If side size is proportionate to the bankroll, it becomes the percentage of side BEYOND THE MINIMUM NUMBER TO ACHIEVE THE CORRECT SIZE
hands_played_by_counters_perc = 0.3
hands_played_by_punters_without_advantage_perc = 0.4
tie_pay = 9
side_size_proportioned_to_bankroll = True


# Dictionary to map the possible results to the corresponding payments
tie_payouts = {
    'Tie 0': 150,
    'Tie 1': 215,
    'Tie 2': 225,
    'Tie 3': 200,
    'Tie 4': 120,
    'Tie 5': 110,
    'Tie 6': 45,
    'Tie 7': 45,
    'Tie 8': 80,
    'Tie 9': 80
}

# Dictionary to map card values
card_values = {
    'A': 1, '2': 2, '3': 3, '4': 4, '5': 5,
    '6': 6, '7': 7, '8': 8, '9': 9, 'T': 0,
    'J': 0, 'Q': 0, 'K': 0
}

# Dictionary to map trigger values for counting
trigger_values = {
    'Tie 0': 7,
    'Tie 1': 7,
    'Tie 2': 6,
    'Tie 3': 7,
    'Tie 4': 7,
    'Tie 5': 7,
    'Tie 6': 7,
    'Tie 7': 4,
    'Tie 8': 6,
    'Tie 9': 6
}

# Dictionary to map counting values for each tie result. The order provides the priority of the sides to count
counting_values = { 
    'Tie 7': {'A': 1, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': -8, '8': 2, '9': 1, 'T': 1, 'J': 1, 'Q': 1, 'K': 1},
    'Tie 6': {'A': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': -7, '7': 1, '8': 1, '9': 1, 'T': 1, 'J': 1, 'Q': 1, 'K': 1},
    'Tie 0': {'A': 2, '2': 2, '3': 2, '4': 2, '5': 1, '6': 1, '7': 0, '8': 1, '9': 1, 'T': -3, 'J': -3, 'Q': -3, 'K': -3},
    'Tie 8': {'A': 1, '2': 1, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': -7, '9': 1, 'T': 1, 'J': 1, 'Q': 1, 'K': 1},
    'Tie 9': {'A': 1, '2': 1, '3': 1, '4': 0, '5': 0, '6': 0, '7': 0, '8': 1, '9': -8, 'T': 1, 'J': 1, 'Q': 1, 'K': 1},
    'Tie 5': {'A': 0, '2': -1, '3': -1, '4': 0, '5': -6, '6': 2, '7': 2, '8': 2, '9': 2, 'T': 0, 'J': 0, 'Q': 0, 'K': 0},
    'Tie 4': {'A': -1, '2': 0, '3': 0, '4': -6, '5': 1, '6': 2, '7': 2, '8': 2, '9': 0, 'T': 0, 'J': 0, 'Q': 0, 'K': 0},
    'Tie 2': {'A': -1, '2': -6, '3': 2, '4': 2, '5': 2, '6': 2, '7': 1, '8': 2, '9': 0, 'T': -1, 'J': -1, 'Q': -1, 'K': -1},
    'Tie 3': {'A': -1, '2': -1, '3': -6, '4': 2, '5': 2, '6': 3, '7': 3, '8': 0, '9': 2, 'T': -1, 'J': -1, 'Q': -1, 'K': -1},
    'Tie 1': {'A': -6, '2': 2, '3': 2, '4': 2, '5': 2, '6': 1, '7': 1, '8': 0, '9': 0, 'T': -1, 'J': -1, 'Q': -1, 'K': -1},
}

# Kelly formula: bankroll percentage to bet on each single tie (side bet) when we have the advantage (expected return > 1)
# The right timing to bet is given when the true count is above the trigger value
# Note: This is valid only with 52 cards cutoff
kelly_bankroll_percentage = { 
    'Tie 0': 0.001114551,
    'Tie 1': 0.000467773,
    'Tie 2': 0.000536545,
    'Tie 3': 0.000583976,
    'Tie 4': 0.000949723,
    'Tie 5': 0.000981404,
    'Tie 6': 0.002509317,
    'Tie 7': 0.002426738,
    'Tie 8': 0.001358738,
    'Tie 9': 0.001249004
}


def initialize_settings(external_settings=None):
    """
    Initialize the settings of the simulation. If external_settings is not None, it will override the default settings.
    
    Parameters:
    - external_settings (dict): External settings to override the default settings
    
    """
    global num_decks, initial_stake, num_players, num_punters, max_side_counts_per_player, \
           cutoff, std_cutoff, num_trips, num_sessions, hours_per_session, hands_per_hour, \
           chips_types_list, bet_min_main, bet_max_main, bet_min_side, bet_max_side, \
           exchanges_allowed_per_player_per_session, exchanges_all_in_one, \
           bankroll_punters_wrt_counters, counters_base_bet_percentages, \
           punters_base_bet_percentages, average_size_main_bet, max_limit_bet, \
           punter_base_bet_multiplier, base_bet_std_perc, \
           side_multiplier_wrt_base, punters_playing_sides_perc, sides_perc_punters_bet_on, \
           hands_played_by_counters_perc, hands_played_by_punters_without_advantage_perc, \
           tie_pay, side_size_proportioned_to_bankroll, kelly_multiplier

    if external_settings is not None:
        num_decks = external_settings.get("num_decks", num_decks)
        initial_stake = external_settings.get("initial_stake", initial_stake)
        num_players = external_settings.get("num_players", num_players)
        num_punters = external_settings.get("num_punters", num_punters)
        max_side_counts_per_player = external_settings.get("max_side_counts_per_player", max_side_counts_per_player)
        cutoff = external_settings.get("cutoff", cutoff)
        std_cutoff = external_settings.get("std_cutoff", std_cutoff)
        num_trips = external_settings.get("num_trips", num_trips)
        num_sessions = external_settings.get("num_sessions", num_sessions)
        hours_per_session = external_settings.get("hours_per_session", hours_per_session)
        hands_per_hour = external_settings.get("hands_per_hour", hands_per_hour)
        chips_types_list = external_settings.get("chips_types_list", chips_types_list)
        bet_min_main = external_settings.get("bet_min_main", bet_min_main)
        bet_max_main = external_settings.get("bet_max_main", bet_max_main)
        bet_min_side = external_settings.get("bet_min_side", bet_min_side)
        bet_max_side = external_settings.get("bet_max_side", bet_max_side)
        exchanges_allowed_per_player_per_session = external_settings.get("exchanges_allowed_per_player_per_session", exchanges_allowed_per_player_per_session)
        exchanges_all_in_one = external_settings.get("exchanges_all_in_one", exchanges_all_in_one)
        bankroll_punters_wrt_counters = external_settings.get("bankroll_punters_wrt_counters", bankroll_punters_wrt_counters)
        counters_base_bet_percentages = external_settings.get("counters_base_bet_percentages", counters_base_bet_percentages)
        punters_base_bet_percentages = external_settings.get("punters_base_bet_percentages", punters_base_bet_percentages)
        average_size_main_bet = external_settings.get("average_size_main_bet", average_size_main_bet)
        max_limit_bet = external_settings.get("max_limit_bet", max_limit_bet)
        punter_base_bet_multiplier = external_settings.get("punter_base_bet_multiplier", punter_base_bet_multiplier)
        base_bet_std_perc = external_settings.get("base_bet_std_perc", base_bet_std_perc)
        side_multiplier_wrt_base = external_settings.get("side_multiplier_wrt_base", side_multiplier_wrt_base)
        punters_playing_sides_perc = external_settings.get("punters_playing_sides_perc", punters_playing_sides_perc)
        sides_perc_punters_bet_on = external_settings.get("sides_perc_punters_bet_on", sides_perc_punters_bet_on)
        hands_played_by_counters_perc = external_settings.get("hands_played_by_counters_perc", hands_played_by_counters_perc)
        hands_played_by_punters_without_advantage_perc = external_settings.get("hands_played_by_punters_without_advantage_perc", hands_played_by_punters_without_advantage_perc)
        tie_pay = external_settings.get("tie_pay", tie_pay)
        side_size_proportioned_to_bankroll = external_settings.get("side_size_proportioned_to_bankroll", side_size_proportioned_to_bankroll)
        kelly_multiplier = external_settings.get("kelly_multiplier", kelly_multiplier)
        

# Extract the side bet list (single ties) from the trigger_values dictionary
sides = list(trigger_values.keys())


def rank_by_role(player):
    return 2 if player.role == "punter" else 1


def assign_sides_to_players():
    """Assign sides to players. 
    
    The number of sides assigned to each player is proportional to the number of sides he can count.
    
    For example, if a player can count 2 sides, he will be assigned 2 sides.
    If there are 4 players and 10 sides, each player will be assigned 2 sides.
    If there are 4 players and 8 sides, each player will be assigned 2 sides.
    If there are 4 players and 9 sides, each player will be assigned 2 sides, except for one player who will be assigned 3 sides.
    
    """
    # Count optimal number of sides per player
    list_of_sides = list(counting_values.keys())
    number_of_sides = len(list_of_sides)
    total_countable_sides = num_players * max_side_counts_per_player

    if total_countable_sides > number_of_sides:
        if (number_of_sides % num_players) == 0:  # if there is no change
            for player in players:
                player.num_assigned_sides = number_of_sides / num_players
        else:
            sorted_players = sorted(players, key=rank_by_role)
            side_per_punter = number_of_sides // num_players
            change = number_of_sides % num_players
            for player in sorted_players:
                if change > 0:
                    player.num_assigned_sides = side_per_punter + 1
                    change -= 1
                else:
                    player.num_assigned_sides = side_per_punter

    else:
        for player in players:
            player.num_assigned_sides = max_side_counts_per_player

    # Sides distribution to the players
    current_index = 0
    for player in players:
        player.sidebets_assigned = list_of_sides[current_index: current_index + player.num_assigned_sides]
        current_index += player.num_assigned_sides
        for side in player.sidebets_assigned:
            player.running_count_per_side[side] = 0
        # print(f"Assigned to {player.name} the sides {player.sidebets_assigned}")


def position_cutoff(cutoff, std_cutoff):
    """
    Params
    ------
    cutoff: int
        The cutoff position
    std_cutoff: int
        The standard deviation of the cutoff position
    
    Return
    ------
    The position of the cutoff card in the shoe
    """
    # Calculate the cutoff based on normal distribution
    not_rounded_cutoff = random.normalvariate(cutoff, std_cutoff)

    # Round to the closest integer
    rounded_cutoff = round(not_rounded_cutoff)

    # Make sure the cutoff is within a valid range
    final_cutoff = max(6, min(rounded_cutoff, num_decks*52))

    return final_cutoff


# Function to generate a shuffled shoe of cards
def generate_shoe():
    # Create a single deck of cards
    single_deck = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'] * 4
    
    # Create the shoe by repeating the single deck for the specified number of decks
    shoe = single_deck * num_decks
    
    # Use SystemRandom to shuffle the shoe for better randomness
    rng = random.SystemRandom()
    rng.shuffle(shoe)
    
    return shoe


def baccarat_score(hand):
    score = sum(card_values[card] for card in hand) % 10
    return score


shoe = generate_shoe()
initial_shoe_lenght = len(shoe)


# Function to get a randomized bet amount based on the average size
def get_randomized_bet(avg_size):
    # Generate a random value using Gaussian distribution with mean as the average size and standard deviation as a percentage of the average size
    randomized_value = random.gauss(avg_size, base_bet_std_perc * avg_size)
    return randomized_value


def round_to_nearest_multiple(value, multiple):
    """Round a value to the nearest multiple of the given base."""
    return round(value/multiple) * multiple


def determine_counter_bet(player):
    # Generate a randomized bet amount based on the average size
    randomized_bet = get_randomized_bet(average_size_main_bet)
    
    # Round the randomized bet to the nearest multiple of the minimum bet
    rounded_randomized_bet = round_to_nearest_multiple(randomized_bet, bet_min_main)
    
    # Ensure the rounded bet is not below the minimum bet
    rounded_randomized_bet = max(rounded_randomized_bet, bet_min_main)
    
    # Randomly choose the betting decision from the available options
    decision = random.choices(['Banker', 'Player', 'Tie'], counters_base_bet_percentages)[0]
    
    # Determine the final bet amount, considering the maximum limit and the rounded bet
    bet_amount = min(max_limit_bet, rounded_randomized_bet, bet_max_main)
    
    # Ensure the bet amount is never 0
    if bet_amount <= 0:
        print(f"{player.name} is attempting to place a bet of {bet_amount} euros.")
    else:
        # Assign the bet amount and decision to the player
        player.bet_amount = bet_amount
        player.bet_choice = decision


def get_counters_to_play():
    """
    Determine which counters will place a bet.
    """
    playing_counters = []

    # For each counter, determine whether they will place a bet or not
    for player in counters:
        # Randomly decide whether the counter will play a hand or not
        play = np.random.binomial(1, hands_played_by_counters_perc)
        if play:
            playing_counters.append(player)

    return playing_counters


def determine_counters_bets():
    # counters = sorted(counters, key=lambda x: x.bankroll, reverse=True)
    # We do not order them because the counters that bet are randomly selected
    playing_counters = get_counters_to_play()
    for player in playing_counters:
        determine_counter_bet(player)

    return counters


# Calculate the average size of the main bet per punter
average_size_main_bet_punter = average_size_main_bet * punter_base_bet_multiplier

# Calculate the size of each side based on the average size of the main bet per punter
size_media_side = average_size_main_bet_punter * side_multiplier_wrt_base

# Calculate the average size per side to bet on using the Kelly criterion
avg_size_per_side_to_bet = {
    side: min(kelly_bankroll_percentage[side] * initial_stake * kelly_multiplier, bet_max_side * num_punters)
    for side in kelly_bankroll_percentage.keys()
}

# If the side size is proportionate to the bankroll, round the average side size per bet
# to the nearest multiple of the minimum side bet, ensuring it is not below the minimum side bet
if side_size_proportioned_to_bankroll:
    rounded_avg_side_size_per_bet = {
        side: max(round_to_nearest_multiple(avg_size_per_side_to_bet[side], bet_min_side), bet_min_side)
        for side in kelly_bankroll_percentage.keys()
    }
else:
    # When the amount per side is not fixed, consider it as if it was always the maximum bet size per side
    rounded_avg_side_size_per_bet = {side: size_media_side * num_punters for side in kelly_bankroll_percentage.keys()}


def get_punters_to_play(signaled_sides):
    """
    Determine which punters will play the hand and which will not.
    Params
    ------
    signaled_sides: list
        The sides signaled by the counters
    
    Return
    ------
    playing_punters: list
        The punters who will play
    min_num_playing_punters: int
        The minimum number of punters who will play
    total_money_per_side: dict
        The total amount of money to be bet on each side
    """
    playing_punters = []
    # For punter, it determines if he will play the hand or not
    # Minimum number of punters is equal to the number of punters needed to cover the maximum amount to be bet for a single side
    # In the case of a non-fixed amount for sides, all punters play (it is considered that potentially everyone will come to bet on that side)
    total_money_per_side = {side: rounded_avg_side_size_per_bet[side] for side in signaled_sides}
    if total_money_per_side:
        min_num_playing_punters = math.ceil(max(total_money_per_side.values()) / bet_max_side)
    else:
        min_num_playing_punters = 0
    for player in punters[:min_num_playing_punters]:
        playing_punters.append(player)
    for player in punters[min_num_playing_punters:]:
        play = np.random.binomial(1, hands_played_by_punters_without_advantage_perc)
        if play:
            playing_punters.append(player)

    return playing_punters, min_num_playing_punters, total_money_per_side


def determine_punters_main_bets_and_amounts(signaled_sides):
    # Get the list of playing punters, minimum number of playing punters, and total money per side
    playing_punters, min_num_playing_punters, total_money_per_side = get_punters_to_play(signaled_sides)
    
    # Iterate over each playing punter
    for player in playing_punters:
        # Generate a randomized bet amount based on the average size of the main bet per punter
        randomized_bet = get_randomized_bet(average_size_main_bet_punter)
        
        # Round the randomized bet to the nearest multiple of the minimum main bet
        rounded_randomized_bet = round_to_nearest_multiple(randomized_bet, bet_min_main)
        
        # Ensure the rounded bet is not below the minimum main bet
        rounded_randomized_bet = max(rounded_randomized_bet, bet_min_main)
        
        # Limit the bet amount to the maximum limit bet and the rounded randomized bet
        bet_amount = min(max_limit_bet, rounded_randomized_bet, bet_max_main)
        
        # Randomly decide the bet choice (Banker, Player, or Tie) based on the punters' base bet percentages
        decision = random.choices(['Banker', 'Player', 'Tie'], punters_base_bet_percentages)[0]
        # print(f"{player.name} decided the amount for te main bet to be {bet_amount}")
        
        # Print the punter's name and the decided amount for the main bet
        # Uncomment the line below if you want to print the information
        # print(f"{player.name} decided the amount for the main bet to be {bet_amount}")
        
        # Ensure the bet amount is never 0
        if bet_amount <= 0:
            print(f"{player.name} is attempting to place a bet of {bet_amount} euros.")
        else:
            # Set the punter's bet amount and bet choice
            player.bet_amount = bet_amount
            player.bet_choice = decision
    
    return playing_punters, min_num_playing_punters, total_money_per_side


# Function to decide sides and amount per side for each punter for all punters
def decide_sides_and_amount_per_side_for_each_punter_for_all_punters(signaled_sides, playing_punters, min_num_playing_punters, total_money_per_side):
    num_active_sides = len(signaled_sides)
    # print(f"Signaled sides: {signaled_sides}")

    sides_with_bets = set()  # track which sides have received bets

    if num_active_sides > 0:
        # Create the precalculated combinations for this iteration
        precalculated_combinations = {}
        for i in range(1, len(signaled_sides)+1):
            precalculated_combinations[i] = list(combinations(signaled_sides, i))
        # print(f"Precalculated combinations: {precalculated_combinations}")

        # Sort punters by bankroll
        punters_sorted = sorted(playing_punters, key=lambda x: x.bankroll, reverse=True)

        # Binomial deviation to decide how many punters are actually playing
        num_punters_playing = min_num_playing_punters + np.random.binomial(len(playing_punters) - min_num_playing_punters, punters_playing_sides_perc)
        punters_that_will_bet = punters_sorted[:num_punters_playing]

        if side_size_proportioned_to_bankroll:
            needed_money_per_side = {side: rounded_avg_side_size_per_bet[side] for side in signaled_sides}

            # Calculate the minimum number of playing punters per side
            min_num_playing_punters_per_side = {side: math.ceil(total_money_per_side[side] / bet_max_side) for side in signaled_sides}
            # print(f"Min num punters playing: {min_num_playing_punters}, where maximum bet to make is: {max(total_money_per_side.values())}")
            for player in punters_that_will_bet:
                # Determine which sides must play minimum
                which_must_play_min = [side for side in min_num_playing_punters_per_side if num_punters_playing == min_num_playing_punters_per_side[side]]
                # print(f"Which ones are to play minimum: {which_must_play_min}")
                how_many_must_play_min = len(which_must_play_min)

                # Assign number of active sides for each punter who will play
                num_sides_this_punter_will_play = how_many_must_play_min + np.random.binomial(num_active_sides-how_many_must_play_min, sides_perc_punters_bet_on)
                # print(f"How many will he play: {num_sides_this_punter_will_play}")
                # if the number of sides to bet on is 0, than leave
                if num_sides_this_punter_will_play <= 0:
                    num_punters_playing -= 1
                else:
                    filtered_combinations = [comb for comb in precalculated_combinations[num_sides_this_punter_will_play] if all(side in comb for side in which_must_play_min)]

                    # If the current list of combinations for num_sides_for_play is empty (because all have been used), refresh it
                    if not filtered_combinations:
                        precalculated_combinations[num_sides_this_punter_will_play] = list(combinations(signaled_sides, num_sides_this_punter_will_play))
                        filtered_combinations = [comb for comb in precalculated_combinations[num_sides_this_punter_will_play] if all(side in comb for side in which_must_play_min)]

                    # Randomly select a combination for this punter
                    chosen_combination = random.choice(filtered_combinations)
                    precalculated_combinations[num_sides_this_punter_will_play].remove(chosen_combination)
                    # print(f"Chosen combination: {chosen_combination}")

                    # Assign the chosen combination to the player's side bets
                    for side in chosen_combination:
                        player.sidebets_chosen_and_relative_amounts[side] = None
                        min_num_playing_punters_per_side[side] -= 1

                    num_punters_playing -= 1
                    # print(f"Number of punters playing goes from {num_punters_playing + 1} to {num_punters_playing}")

            # Count how many punters have a side following and allocate the amount equally
            punters_on_this_side = {side: 0 for side in signaled_sides}
            for player in punters_that_will_bet:
                for side in player.sidebets_chosen_and_relative_amounts.keys():
                    punters_on_this_side[side] += 1
                # print(f"Punters on this side: {punters_on_this_side}")
            # print(f"Necessary money still on this side: {needed_money_per_side}")
            for player in punters_that_will_bet:
                for side in player.sidebets_chosen_and_relative_amounts:
                    # Determine the amount to bet on each side
                    bet = round_to_nearest_multiple(needed_money_per_side[side]/punters_on_this_side[side], bet_min_side)
                    side_amount = min(bet, bet_max_side)
                    player.sidebets_chosen_and_relative_amounts[side] = side_amount
                    needed_money_per_side[side] -= side_amount
                    punters_on_this_side[side] -= 1

            # Check that the total amount to be bet on each side matches the necessary money per side
            for side in punters_on_this_side.keys():
                if not total_money_per_side[side] == sum(player.sidebets_chosen_and_relative_amounts.get(side, 0) for player in punters_that_will_bet):
                    print("ERROR: total amount to be bet on a side other than the amount established by kelly formula")
                    print(f"On side {side} the total money should be {total_money_per_side[side]} but it is {sum(player.sidebets_chosen_and_relative_amounts.get(side, 0) for player in punters_that_will_bet)}")
                    raise
        else:
            for player in punters_that_will_bet:
                # Assign number of active sides for each punter who will play
                num_sides_this_punter_will_play = np.random.binomial(num_active_sides, sides_perc_punters_bet_on)

                # If the number of sides to bet on is greater than 0
                if num_sides_this_punter_will_play > 0:
                    # If the current list of combinations for num_sides_for_play is empty (because all have been used), refresh it
                    if not precalculated_combinations[num_sides_this_punter_will_play]:
                        precalculated_combinations[num_sides_this_punter_will_play] = list(combinations(signaled_sides, num_sides_this_punter_will_play))

                    # Randomly select a combination for this punter
                    chosen_combination = random.choice(precalculated_combinations[num_sides_this_punter_will_play])
                    precalculated_combinations[num_sides_this_punter_will_play].remove(chosen_combination)

                    # Assign the chosen combination to the player's side bets
                    for side in chosen_combination:
                        player.sidebets_chosen_and_relative_amounts[side] = None
                        sides_with_bets.add(side)

            # Ensure each side gets at least one bet
            sides_without_bets = set(signaled_sides) - sides_with_bets
            if sides_without_bets:
                for side in sides_without_bets:
                    random.choice(punters_that_will_bet).sidebets_chosen_and_relative_amounts[side] = None

            for player in punters_that_will_bet:
                # Determine the amount to bet on each side
                randomized_bet = get_randomized_bet(size_media_side)
                rounded_randomized_bet = round_to_nearest_multiple(randomized_bet, bet_min_side)
                rounded_randomized_bet = max(rounded_randomized_bet, bet_min_side)
                side_amount = min(rounded_randomized_bet, bet_max_side)

                for side in player.sidebets_chosen_and_relative_amounts:
                    player.sidebets_chosen_and_relative_amounts[side] = side_amount


# Determine the bets for the punters based on the signaled sides
def determine_punters_bets(signaled_sides):
    playing_punters, min_num_playing_punters, total_money_per_side = determine_punters_main_bets_and_amounts(signaled_sides)
    decide_sides_and_amount_per_side_for_each_punter_for_all_punters(signaled_sides, playing_punters, min_num_playing_punters, total_money_per_side)


# Determine the bets for all players
def determine_all_players_bets(signaled_sides):
    determine_punters_bets(signaled_sides)
    determine_counters_bets()

    for player in players:
        player.total_bet_amount = player.bet_amount + sum(player.sidebets_chosen_and_relative_amounts.values())


# Place bets for all players and adjust their bankroll
def players_place_bets(session_summary):
    total_bet_on_every_bet_type = {side: 0 for side in ['Banker', 'Player', 'Tie'] + list(counting_values)}
    initial_settings_counter_hand = 0

    # Check if players can place bets and handle bankruptcy
    for player in players:
        can_place_bets, reason, initial_values_assignment = handle_bankruptcy(player, session_summary)
        initial_settings_counter_hand += initial_values_assignment
        # if player.total_bet_amount != (player.bet_amount + sum(player.sidebets_chosen_and_relative_amounts.values())):
            # print("TOTAL BET AMOUNT DIFFERENT FROM THE SUM OF BETS AMOUNT")
        if not can_place_bets:
            if reason == "bankruptcy":
                # print("bankruptcy inside PLAYERS PLACE BETS")
                return True, "bankruptcy", None, None, initial_settings_counter_hand
            else:
                if player.role == 'punter':
                    return True, "punter_bankruptcy", None, None, initial_settings_counter_hand
                else:
                    return True, "counter_bankruptcy", None, None, initial_settings_counter_hand

    # Place bets and adjust bankroll for each player
    for player in players:
        # print(f"{player.name} bets {player.bet_amount} on {player.bet_choice} passing from {player.bankroll} to {player.bankroll-player.total_bet_amount}")
        if player.bankroll >= player.total_bet_amount:
            player.bankroll -= player.total_bet_amount
        else:
            print("BANKROLL GOES NEGATIVE IN PLAYERS PLACE BETS: The player tries to place a bet that is larger than his bankroll")
            # if player.total_bet_amount != (player.bet_amount + sum(player.sidebets_chosen_and_relative_amounts.values())):
                # print("TOTAL BET AMOUNT DIFFERENT FROM THE SUM OF BETS AMOUNT")
    total_bet_for_this_hand = sum(player.total_bet_amount for player in players)
    for player in players:
        if player.bet_choice is not None:
            total_bet_on_every_bet_type[player.bet_choice] += player.bet_amount
        for side in player.sidebets_chosen_and_relative_amounts:
            total_bet_on_every_bet_type[side] += player.sidebets_chosen_and_relative_amounts[side]
        # print(f"Sum of totals bet on every bet: {sum(total_bet_on_every_bet_type.values())}")

    return False, None, total_bet_for_this_hand, total_bet_on_every_bet_type, initial_settings_counter_hand


# Function to resolve bets and adjust player's bankroll based on the outcome
def dealer_resolves_bets(outcome):
    """
    Place the bets and adjust the player's bankroll based on the outcome of the main bet and side bets.
    """
    total_win_for_each_bet = {side: 0 for side in ['Banker', 'Player', 'Tie'] + list(counting_values)}
    bankroll_total_before_withdrawal = sum(player.bankroll for player in players)

    for player in players:
        # Adjust bankroll based on main bet
        if player.bet_choice == outcome or player.bet_choice == "Tie" and outcome.startswith("Tie "):
            if outcome == 'Player':
                if player.bet_amount >= 0:
                    player.bankroll += 2 * player.bet_amount
                    player.total_win_for_each_bet['Player'] = 2 * player.bet_amount
                    # print(f"{player.name} bet on Player {player.bet_amount}")
                else:
                    print("BET WON BUT NEGATIVE VALUE PLAYER BET (dealer_resolve_bets)")
            elif outcome == 'Banker':
                if player.bet_amount >= 0:
                    player.bankroll += 1.95 * player.bet_amount
                    player.total_win_for_each_bet['Banker'] = 1.95 * player.bet_amount
                    # print(f"{player.name} bet on Banker {player.bet_amount}")
                else:
                    print("BET WON BUT NEGATIVE VALUE LAY BET (dealer_resolve_bets)")
            else:
                if player.bet_amount >= 0:
                    player.bankroll += (tie_pay + 1) * player.bet_amount
                    player.total_win_for_each_bet['Tie'] = (tie_pay + 1) * player.bet_amount
                    # print (f"{player.name} bet on Tie {player.bet_amount}")
                else:
                    print("BET WON BUT NEGATIVE VALUE DRAW BET (dealer_resolve_bets)")

        # Returns the money to those who bet on players and bankers
        elif outcome is not None and outcome.startswith("Tie "):
            if player.bet_amount >= 0:
                player.bankroll += player.bet_amount
                player.total_win_for_each_bet[player.bet_choice] = player.bet_amount
                # Handle side bets:
            else:
                print("BET WON BUT NEGATIVE BET VALUE (dealer_resolve_bets)")

        # Handle side bets
        if outcome is not None and outcome.startswith("Tie "):
            for side, amount in player.sidebets_chosen_and_relative_amounts.items():
                if side == outcome:
                    payout_ratio = tie_payouts[outcome]
                    if amount >= 0:
                        player.bankroll += (payout_ratio + 1) * amount  # +1 to include the original bet amount
                        player.total_win_for_each_bet[side] = (payout_ratio + 1) * amount
                    else:
                        print("BET WON BUT TRIED TO BET NEGATIVE VALUE AMOUNT ON SIDE BET (dealer_resolve_bets)")
        # print(f"[DEBUG] {player.name}'s bankroll after resolving bets: {player.bankroll} euros.")
        # Reset the bets for the player for the next round
    final_bankroll_after_withdrawal = sum(player.bankroll for player in players)
    bankroll_won_on_this_hand = final_bankroll_after_withdrawal - bankroll_total_before_withdrawal

    for player in players:
        if player.bet_choice is not None:
            total_win_for_each_bet[player.bet_choice] += player.total_win_for_each_bet[player.bet_choice]
        for side in player.sidebets_chosen_and_relative_amounts:
            total_win_for_each_bet[side] += player.total_win_for_each_bet[side]
        player.reset_bet()

    return bankroll_won_on_this_hand, final_bankroll_after_withdrawal, total_win_for_each_bet


def calculate_ideal_bankroll_for_players():
    """
    Calculate the ideal bankroll for counters and punters based on the given total stake.
    """
    sum_bankroll = sum(player.bankroll for player in players)
    initial_values_assignment = 0
    if sum_bankroll < 0:
        print("ERROR, sum of players' bankrolls negative")

    if sum_bankroll == 0:
        total_initial_stake = initial_stake
        total_initial_for_check = initial_stake
        # print("Checking Initial Values for the Ideal Bankroll")
        initial_values_assignment += 1
        # for player in players:
            # print(f"At the time of assigning the initial values, the players' bankroll is:")
            # print(f" {player.name}: {player.ideal_bankroll}")
    else:
        total_initial_for_check = sum_bankroll
        total_initial_stake = sum_bankroll - sum(player.total_bet_amount for player in players)
    # print(f"Total initial stake before calculation of ideal redistribution: {total_initial_stake}")

    if num_punters == 0:
        ideal_bankroll_counter = total_initial_stake / num_players
        rounded_ideal_bankroll_counter = round_to_nearest_multiple(ideal_bankroll_counter, min(chips_types_list))
        rounded_ideal_bankroll_punter = bet_min_main

    elif num_punters - num_players == 0:
        ideal_bankroll_punter = total_initial_stake / num_punters
        rounded_ideal_bankroll_punter = round_to_nearest_multiple(ideal_bankroll_punter, min(chips_types_list))
        rounded_ideal_bankroll_counter = bet_min_main

    else:
        ideal_bankroll_counter = total_initial_stake / (
                num_players + (num_punters * (bankroll_punters_wrt_counters - 1)))
        rounded_ideal_bankroll_counter = round_to_nearest_multiple(ideal_bankroll_counter, min(chips_types_list))
        ideal_bankroll_punter = (bankroll_punters_wrt_counters * total_initial_stake) / (
                num_players + (num_punters * (bankroll_punters_wrt_counters - 1)))
        rounded_ideal_bankroll_punter = round_to_nearest_multiple(ideal_bankroll_punter, min(chips_types_list))

    if rounded_ideal_bankroll_counter < 0 or rounded_ideal_bankroll_punter < 0:
        # print(f"Ideal bankroll counter NON-ROUNDED: {ideal_bankroll_counter}")
        # print(f"Ideal per counter: {rounded_ideal_bankroll_counter}")
        # print(f"Ideal bankroll punter (depends on counter) NON-ROUNDED: {ideal_bankroll_punter}")
        # print(f"Ideal per punter: {rounded_ideal_bankroll_punter}")
        # print(f"Total initial stake: {total_initial_stake}")
        # total_ideal_hypothesis = (rounded_ideal_bankroll_punter * num_punters) + (rounded_ideal_bankroll_counter * (num_players - num_punters))
        # print(f"Total of ideal hypothesis: {total_ideal_hypothesis}")
        # print("I'm going bankruptcy into the ideal bankroll calculation function because the redistribution gives me a rounded_ideal_bankroll (or punters or counters) less than the minimum bet size")
        return "bankruptcy", initial_values_assignment

    for player in players:
        if player.role == "punter":
            player.ideal_bankroll = rounded_ideal_bankroll_punter+player.total_bet_amount
            # print(f"Ideal for {player.name}: {player.ideal_bankroll}")
        elif player.role == "counter":
            player.ideal_bankroll = rounded_ideal_bankroll_counter+player.total_bet_amount
            # print(f"Ideal for {player.name}: {player.ideal_bankroll}")

    # Step 2: Check if the sum of bankrolls matches the initial total bankroll
    total_after_rounding = sum(player.ideal_bankroll for player in players)
    discrepancy = total_initial_for_check - total_after_rounding
    # print(f"Total of ideal hypothesis in ideal bankrolls: {total_after_rounding}")

    if discrepancy != 0:
        if discrepancy > (num_players * bet_min_main):
            print(f"ERROR: Discrepancy higher than expected ({discrepancy})")
            print(f"Initial stake: {total_initial_for_check}")
            print(f"Ideal stake calculated and rounded: {total_after_rounding}")
            for player in players:
                print(f"Bankroll of {player.name}: {player.bankroll}")
                print(f"Ideal bankroll calculated for {player.name}: {player.ideal_bankroll}")
                # player.ideal_bankroll = player.old_bankroll
        elif discrepancy < 0:
            sorted_players = sorted(players, key=lambda player: player.ideal_bankroll - player.total_bet_amount, reverse=True)
            discrepancy_solved = 0
            for player in sorted_players:
                amount = min(player.ideal_bankroll - player.total_bet_amount, - discrepancy - discrepancy_solved)
                player.ideal_bankroll -= amount
                discrepancy_solved += amount
                if discrepancy_solved == -discrepancy:
                    break
        else:  # if discrepancy is positive
            sorted_players = sorted(players, key=lambda player: player.ideal_bankroll - player.total_bet_amount)
            for player in sorted_players:
                player.ideal_bankroll += discrepancy
                break

    # Ensure the total bankroll remains unchanged
    total_final = sum(player.ideal_bankroll for player in players)
    if total_initial_for_check != total_final:
        print(f"Ideal bankroll calculation mismatch!")
        print(f"Initial: {total_initial_for_check}, Final: {total_final}")
        print(f"Ideal bankroll calculated for the counters: {rounded_ideal_bankroll_counter}")
        print(f"Ideal bankroll calculated for punters: {rounded_ideal_bankroll_punter}")
        for player in players:
            print(f"Bankroll of {player.name}: {player.bankroll}")
            print(f"Ideal bankroll calculated for {player.name}: {player.ideal_bankroll}")
    # print(f"I have finished hypothesizing a redistribution")

    for player in players:
        if not player.ideal_bankroll >= player.total_bet_amount:
            print("MISTAKE: Inside calculate_ideal_bankroll ideal BANKROLL WOULD GO NEGATIVE by subtracting the total_bet_amount: the player tries to place a bet larger than his bankroll")
            print(f"{player.name} would have {player.ideal_bankroll} and would have to bet {player.total_bet_amount}")
            print(f"Discrepancy: {discrepancy}")

    return None, initial_values_assignment


def assign_all_ideal_bankrolls():
    # Calculate the ideal bankroll for each player
    result, initial_values_assignment = calculate_ideal_bankroll_for_players()

    if result is None:
        # If the calculation is successful, assign the ideal bankroll to each player
        for player in players:
            if player.ideal_bankroll >= 0:
                player.bankroll = player.ideal_bankroll
            else:
                print("ERROR: Negative Ideal Player Bankroll, Do Not Assign It")
            # Uncomment the following lines for debugging purposes
            # print(f"{player.name}: Old Bankroll = {player.old_bankroll}")
            # print(f"{player.name}: Bankroll = {player.bankroll}")
    else:
        # If the calculation fails, indicate bankruptcy and return the initial values assignment
        # Uncomment the following line for debugging purposes
        # print("I'm going bankruptcy in AIO redistribution because the ideal bankroll calculation function tells me")
        return "bankruptcy", initial_values_assignment

    return None, initial_values_assignment


def redistribute_bankroll_single_player(bankrupt_player):
    """
    Redistributes bankroll to bring the bankrupt player back to the game.
    Takes money from the player(s) who have the highest positive deviation from their ideal percentage.

    Params
    ------
    bankrupt_player: Player
        The player who is going bankrupt
    
    Return
    ------
    donor: Player
        The player who is donating money to the bankrupt player
    bankrupt_player: Player
        The player who is going bankrupt
    reason: str
        The reason for the bankruptcy
    """
    result, initial_values_assignment = calculate_ideal_bankroll_for_players()
    if result is None:
        # Calculate the amount required for the bankrupt player
        required_amount = bankrupt_player.ideal_bankroll - bankrupt_player.bankroll
        possible_players = players.copy()
        possible_players.remove(bankrupt_player)
        possible_donors_dict = {}

        for player in possible_players:
            if player.bankroll - player.ideal_bankroll > 0:
                possible_donors_dict[player.name] = player.bankroll - player.ideal_bankroll
        sorted_possible_donors_dict = {bettor: transferable_cash for bettor, transferable_cash in sorted(possible_donors_dict.items(), key=lambda item: item[1], reverse=True)}

        if required_amount > 0:

            if not sorted_possible_donors_dict:  # No suitable donor available
                return None, None, "bankruptcy"

            else:
                donor_name = next(iter(sorted_possible_donors_dict))
                donor = next(player for player in players if player.name == donor_name)
                transfer_amount = min(required_amount, sorted_possible_donors_dict[donor_name])
                
                # Transfer the rounded amount
                donor.bankroll -= transfer_amount
                bankrupt_player.bankroll += transfer_amount
                # print(f"[DEBUG] {donor.name}'s bankroll after transferring: {donor.bankroll}")
                # print(f"[DEBUG] {bankrupt_player.name}'s bankroll after receiving: {bankrupt_player.bankroll}")
                # print("Finished redistribution between individuals")
                # print(f"Donors: {donor.name}")
                # print(f"Player who was going bankrupt: {bankrupt_player.name}")
                
                if donor.bankroll < donor.total_bet_amount:
                    print(f"ERROR: Inside redistribute_bankroll_single_player donor's BANKROLL WOULD GO NEGATIVE by subtracting the total_bet_amount: the player {donor.name} He would try to place a bet bigger than his bankroll ({donor.total_bet_amount})")
                    print(f"The list of possible donors is:")
                    print(f"Bankrupt player: {bankrupt_player.name}, bankroll {bankrupt_player.bankroll} e total bet amount {bankrupt_player.total_bet_amount}")
                    for player in players:
                        if player.name in possible_donors_dict:
                            print(f"{player.name} who has a bankroll of {player.bankroll} and an ideal bankroll of {player.ideal_bankroll}")
                    print(f"The system chose as a donor: {donor.name}")

                return donor, bankrupt_player, None

        elif required_amount == 0:
            return None, None, "bankruptcy"
        else:
            """
            print(f"Negative Trade Request Amount: {required_amount}")
            print(f"Ideal bankroll of bankruptcy player: ({bankrupt_player.ideal_bankroll}")
            print(f"Bankroll of bankrupt player: ({bankrupt_player.bankroll})")
            print(
                f"Amount requested from a potential donor: ({bankrupt_player.ideal_bankroll - bankrupt_player.bankroll})")
            """
            for player in players:
                print(f"{player.name}: Bankroll = {player.bankroll}")

            print("I'm going bankruptcy in single redistribution because negative exchange amount")
            print(f"{bankrupt_player.name} is trying to ask {required_amount}")
            return None, None, "bankruptcy"
    else:
        # print("I'm going bankruptcy in single redistribution because the calculation of the ideal bankroll tells me so")
        return None, None, "bankruptcy"


# 2. Modification of handle_bankruptcy function to update session_summary based on failure reason
def handle_bankruptcy(player, session_summary):
    """
    Checks if a player has enough bankroll to cover their bets.
    If not, attempts to redistribute bankroll.
    """
    initial_values_assignment = 0
    if sum(player.bankroll for player in players) < sum(player.total_bet_amount for player in players):
        return False, "bankruptcy", initial_values_assignment
    # Check if player has sufficient bankroll
    if player.bankroll >= player.total_bet_amount:
        return True, None, initial_values_assignment

    # If player doesn't have enough bankroll, check if they can use exchanges
    while player.exchanges_used < exchanges_allowed_per_player_per_session:
        if exchanges_all_in_one:
            result_aio, initial_values_assignment = assign_all_ideal_bankrolls()
            if result_aio is not None:
                for player in players:
                    player.exchanges_used += 1  # Increment the exchanges used for all players
                    if player.bankroll >= player.total_bet_amount:
                        return True, None, initial_values_assignment
            else:
                sum_bankroll_during_bankruptcy = sum(player.bankroll for player in players)
                rounding_remainder_punters = min(chips_types_list) * num_punters
                rounding_remainder_counters = min(chips_types_list) * (num_players - num_punters)
                sum_of_bets_to_be_made_this_round = sum(player.total_bet_amount for player in players)
                max_remainder = rounding_remainder_punters + rounding_remainder_counters + sum_of_bets_to_be_made_this_round
                if sum_bankroll_during_bankruptcy > max_remainder:
                    print("bankruptcy NOT JUSTIFIED")
                    print(f"Total bankrolls sum: {sum_bankroll_during_bankruptcy}")
                    print(f"max_remainder : {max_remainder}")
                # print("I go bankruptcy INSIDE HANDLE BANKRUPTCY FUNCTION")
                # print("--------------bankruptcy--------------")
                return False, "bankruptcy", initial_values_assignment

        else:
            donor, bankrupted_player, result_if_bankruptcy = redistribute_bankroll_single_player(player)
            if result_if_bankruptcy is None:
                bankrupted_player.exchanges_used += 1  # Increment the exchanges used for the bankrupt player
                donor.exchanges_used += 1  # Increment the exchanges used for the donor
                if player.bankroll >= player.total_bet_amount:
                    return True, None, initial_values_assignment
            else:
                sum_bankroll_during_bankruptcy = sum(player.bankroll for player in players)
                rounding_remainder_punters = min(chips_types_list) * num_punters * 3
                rounding_remainder_counters = (min(chips_types_list) * (num_players - num_punters))
                sum_of_bets_to_be_made_this_round = sum(player.total_bet_amount for player in players)
                max_remainder = rounding_remainder_punters + rounding_remainder_counters + sum_of_bets_to_be_made_this_round
                if sum_bankroll_during_bankruptcy > max_remainder:
                    print("bankruptcy NOT JUSTIFIED")
                    print(f"Total bankrolls sum: {sum_bankroll_during_bankruptcy}")
                    print(f"max_remainder : {max_remainder}")
                # print("I go bankruptcy INSIDE HANDLE BANKRUPTCY FUNCTION")
                # print("--------------bankruptcy--------------")
                # for player in players:
                #    print(f"Bankroll of {player.name}: {player.bankroll}")
                # print("---------------------------------------")
                # print()
                return False, "bankruptcy", initial_values_assignment

    # If player has no available exchanges, mark failure
    if player.role == "punter":
        session_summary["punter_failures"] += 1
    else:
        session_summary["counter_failures"] += 1
    # print(f"Session ending early due to failure of {player.name}, he ran out of redistributions.")

    return False, "Run out of exchanges", initial_values_assignment


def determine_hot_sides(shoe_length):
    """
    Determines the sides that are considered "hot" based on the running count per side of each player.
    If the count is above the trigger value for a side, it is added to the list of signaled sides.
    Returns a list of hot sides.
    """
    signaled_sides = []
    num_remaining_decks_in_the_shoe = shoe_length / 52

    for player in players:
        for side, count in player.running_count_per_side.items():
            if count / num_remaining_decks_in_the_shoe >= trigger_values[side]:
                signaled_sides.append(side)

    hot_sides = list(set(signaled_sides))  # Remove duplicates

    return hot_sides


# Simulate a single hand of baccarat
def simulate_hand(shoe, money_at_last_hand, session_summary):
    # Reset the bets for all players
    for player in players:
        player.reset_bet()
    
    # Calculate the total money at the start of the hand
    money_at_start_hand = sum(player.bankroll for player in players)
    
    # Check for any discrepancy in money from the last hand to this hand
    if money_at_last_hand != money_at_start_hand:
        print("ERROR: Money discrepancy from the last hand to this hand")
    
    # Determine the sides that are considered "hot" based on the shoe length
    shoe_length = len(shoe)
    signaled_sides = determine_hot_sides(shoe_length)
    
    # Determine the bets for all players based on the signaled sides
    determine_all_players_bets(signaled_sides)
    
    # Place bets for all players and check if bankruptcy occurred
    bankruptcy_occurred, failure_reason, total_bet_for_this_hand, total_bet_on_every_bet_type, initial_settings_counter_hand = players_place_bets(session_summary)
    
    # Calculate the total money after this hand
    money_after_this_hand = sum(player.bankroll for player in players)
    
    # Check if the money at the end of the hand is less than the money at the beginning of the hand
    # but no bets were placed
    if money_after_this_hand < money_at_start_hand and total_bet_for_this_hand == 0:
        print("ERROR: Money at the end of the hand is less than money at the beginning of the hand but no bets were placed")
    
    # If bankruptcy occurred, return the necessary information
    if bankruptcy_occurred:
        return None, bankruptcy_occurred, failure_reason, money_after_this_hand, total_bet_for_this_hand, None, total_bet_on_every_bet_type, None, initial_settings_counter_hand, None
    
    # Deal the initial hands for the player and the banker
    player_hand = [shoe.pop(), shoe.pop()]
    banker_hand = [shoe.pop(), shoe.pop()]
    
    # Calculate the baccarat score for the player and the banker
    baccarat_score_player = baccarat_score(player_hand)
    baccarat_score_banker = baccarat_score(banker_hand)
    
    # Determine the actions based on the baccarat scores, following baccarat rules
    if baccarat_score_player in [8, 9] or baccarat_score_banker in [8, 9]:
        pass
    elif baccarat_score_player <= 5:
        player_hand.append(shoe.pop())
        
        if baccarat_score_banker <= 2:
            banker_hand.append(shoe.pop())
        elif baccarat_score_banker == 3 and player_hand[2] != '8':
            banker_hand.append(shoe.pop())
        elif baccarat_score_banker == 4 and player_hand[2] in ['2', '3', '4', '5', '6', '7']:
            banker_hand.append(shoe.pop())
        elif baccarat_score_banker == 5 and player_hand[2] in ['4', '5', '6', '7']:
            banker_hand.append(shoe.pop())
        elif baccarat_score_banker == 6 and player_hand[2] in ['6', '7']:
            banker_hand.append(shoe.pop())
    elif baccarat_score_banker <= 5:
        banker_hand.append(shoe.pop())
    
    # Combine the player and banker hands
    cards_on_the_table = player_hand + banker_hand
    
    # Determine the outcome of the hand
    if baccarat_score(player_hand) > baccarat_score(banker_hand):
        outcome = 'Player'
    elif baccarat_score(player_hand) < baccarat_score(banker_hand):
        outcome = 'Banker'
    else:
        outcome = 'Tie ' + str(baccarat_score(player_hand))
    
    # Update the running count for each player based on the cards on the table
    for card in cards_on_the_table:
        for player in players:
            player.update_running_count(card)
    
    # Resolve the bets and calculate the total win in this hand
    total_win_in_this_hand, money_after_this_hand, total_win_for_each_bet = dealer_resolves_bets(outcome)
    
    # Check if any player's bankroll goes negative
    for player in players:
        if player.bankroll < 0:
            print("bankroll of a player goes negative")
    
    # Calculate the total bets on each bet type
    total_bets_each_bet = sum(total_bet_on_every_bet_type.values())
    
    # Check if the total bet for this hand matches the sum of total bets on each bet
    if total_bet_for_this_hand != total_bets_each_bet:
        print(f"Total bet for this hand ({total_bet_for_this_hand}) different from sum total bets on each bet ({total_bets_each_bet})")
    
    # Check if the total money at the end of the hand is correct
    if not round(sum(player.bankroll for player in players)) == round(money_at_last_hand - total_bet_for_this_hand + total_win_in_this_hand):
        print("The money total at the end of the hand is incorrect")
        print(f"Sum of the bankrolls at the moment: {sum(player.bankroll for player in players)}")
        print(f"It should be equal to last hand money: {money_at_last_hand} - total bet on this hand {total_bet_for_this_hand} + total won on this hand {total_win_in_this_hand}")
    
    # Return the outcome and other information
    return outcome, False, None, money_after_this_hand, total_bet_for_this_hand, total_win_in_this_hand, total_bet_on_every_bet_type, total_win_for_each_bet, initial_settings_counter_hand, signaled_sides


# Simulate a session of baccarat
def simulate_session(shoe, money_last_hand, session_summary):
    session_outcomes = []  # List to store the outcomes of each hand
    failure_reasons = []  # List to store the reasons for failure
    hands_to_play = hours_per_session * hands_per_hour  # Calculate the total number of hands to play
    totals_bet = []  # List to store the total bets made in each hand
    totals_won = []  # List to store the total winnings in each hand
    totals_bet_for_each_bet = {}  # Dictionary to store the total bets made for each type of bet
    totals_won_for_each_bet = {}  # Dictionary to store the total winnings for each type of bet
    session_assigning_initial_values = 0  # Variable to track the assigning of initial values
    signaled_sides_session = []  # List to store the signaled sides for the session

    session_summary["total_sessions"] += 1  # Increment the total number of sessions

    for hand_num in range(hands_to_play):
        if len(shoe) <= position_cutoff(cutoff, std_cutoff):
            # Reshuffle the cards if the shoe reaches the cutoff point
            shoe = generate_shoe()
            for player in players:
                # print(player.running_count_per_side)
                for side in player.running_count_per_side:
                    player.running_count_per_side[side] = 0

        # Simulate a single hand and get the outcome, end_early flag, reason, and other statistics
        outcome, end_early, reason, money_finished_hand, total_bet, total_won, total_bet_for_each_bet, total_won_for_each_bet, hand_assigning_initial_values, signaled_sides = simulate_hand(shoe, money_last_hand, session_summary)

        # Update the lists and dictionaries with the statistics from the hand
        totals_bet.append(total_bet)
        totals_won.append(total_won)
        session_outcomes.append(outcome)
        money_last_hand = money_finished_hand
        session_assigning_initial_values += hand_assigning_initial_values

        # Check if signaled_sides is not None and extend signaled_sides_session list
        if signaled_sides is not None:
            signaled_sides_session.extend(signaled_sides)

        if total_bet_for_each_bet is not None:
            for main_or_side, amount in total_bet_for_each_bet.items():
                if main_or_side in totals_bet_for_each_bet:
                    totals_bet_for_each_bet[main_or_side] += amount
                else:
                    totals_bet_for_each_bet[main_or_side] = amount

        if total_bet_for_each_bet is not None:
            for main_or_side, amount in total_won_for_each_bet.items():
                if main_or_side in totals_won_for_each_bet:
                    totals_won_for_each_bet[main_or_side] += amount
                else:
                    totals_won_for_each_bet[main_or_side] = amount

        if reason:
            failure_reasons.append(reason)

        if end_early:  # Check if session should end early
            # print("BANKRUPTCY SESSION")
            break
    else:
        session_summary["completed_sessions"] += 1

    end_of_session_bankrolls = [player.bankroll for player in players]
    return session_outcomes, failure_reasons, end_of_session_bankrolls, money_last_hand, totals_bet, totals_won, totals_bet_for_each_bet, totals_won_for_each_bet, session_assigning_initial_values, signaled_sides_session


def simulate_trip(trip_index, session_summary):
    # Print the start of the trip
    print(f"Start of trip number: {trip_index}")
    
    # Reset bankrolls for all players
    for player in players:
        player.bankroll = 0
    
    # Assign ideal bankrolls to players and get the bankruptcy result and initial values assignment
    bankruptcy_result, initial_values_assignment = assign_all_ideal_bankrolls()
    
    # Initialize variables for trip outcomes, failure reasons, end of trip bankrolls, totals bet, totals won, and signaled sides
    remaining_cards_at_the_end_of_session = []
    trip_outcomes = []
    all_failure_reasons = []
    end_of_the_trip_bankrolls = []
    totals_bet_trip = []
    totals_won_trip = []
    money_start_of_trip = initial_stake
    trip_totals_bet_for_each_bet = {}
    trip_totals_won_for_each_bet = {}
    trip_initial_values_assignment = initial_values_assignment
    signaled_sides_trip = []

    # Iterate over the number of sessions
    for _ in range(num_sessions):
        # Generate a new shoe for each session
        shoe = generate_shoe()
        
        # Simulate the session and get the session outcomes, failure reasons, end of session bankrolls, money at the end of session,
        # totals bet, totals won, totals bet for each bet, totals won for each bet, session assigning initial values, and signaled sides
        session_outcomes, failure_reasons, end_of_session_bankrolls, money_end_of_session, totals_bet, totals_won, totals_bet_for_each_bet, totals_won_for_each_bet, session_assigning_initial_values, signaled_sides_session = simulate_session(shoe, money_start_of_trip, session_summary)
        
        # Append session outcomes, failure reasons, totals bet, totals won, and signaled sides to trip outcomes
        trip_outcomes.extend(session_outcomes)
        all_failure_reasons.extend(failure_reasons)
        totals_bet_trip.extend(totals_bet)
        totals_won_trip.extend(totals_won)
        end_of_the_trip_bankrolls = end_of_session_bankrolls
        remaining_cards_at_the_end_of_session.append(len(shoe))
        money_start_of_trip = money_end_of_session
        trip_initial_values_assignment += session_assigning_initial_values
        signaled_sides_trip.extend(signaled_sides_session)

        # Update trip totals bet for each bet
        for main_or_side, amount in totals_bet_for_each_bet.items():
            if main_or_side in trip_totals_bet_for_each_bet:
                trip_totals_bet_for_each_bet[main_or_side] += amount
            else:
                trip_totals_bet_for_each_bet[main_or_side] = amount

        # Update trip totals won for each bet
        for main_or_side, amount in totals_won_for_each_bet.items():
            if main_or_side in trip_totals_won_for_each_bet:
                trip_totals_won_for_each_bet[main_or_side] += amount
            else:
                trip_totals_won_for_each_bet[main_or_side] = amount

        # Reset session for each player
        for player in players:
            player.reset_session()

        # Check if the last failure reason is bankruptcy
        if failure_reasons and failure_reasons[-1] == "bankruptcy":
            # Increment the bankruptcy count in the session summary
            session_summary["bankruptcies"] += 1
            break
        else:
            # Redistribute bankroll using exchange all-in-one (aio) at the end of the session
            bankruptcy_result, initial_values_assignment_end_session = assign_all_ideal_bankrolls()
            trip_initial_values_assignment += initial_values_assignment_end_session
    return trip_outcomes, all_failure_reasons, end_of_the_trip_bankrolls, remaining_cards_at_the_end_of_session, totals_bet_trip, totals_won_trip, trip_totals_bet_for_each_bet, trip_totals_won_for_each_bet, trip_initial_values_assignment, signaled_sides_trip


# Simulate multiple trips
def simulate_multiple_trips(session_summary):

    # Preparation of the arguments for every time we call a simulate_trip
    trip_params = [(i, session_summary) for i in range(num_trips)]

    # Use of starmap to execute simulate_trip with multiple arguments
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()-2) as pool:
        trips_results = pool.starmap(simulate_trip, trip_params)
        pool.close()
        pool.join()

    # Initialize variables to store trip results
    all_trip_results = []
    all_end_of_trip_bankrolls = []
    all_remaining_cards_at_the_end_of_session = []
    total_bet_total = []
    total_won_total = []
    all_total_bets_each_bet = {}
    all_total_won_each_bet = {}
    all_initial_values_assignment = 0
    totals_end_of_trip_bankroll = []
    signaled_sides_multitrip = []

    # Process trip results
    for i, trip_result in enumerate(trips_results):
        trip_outcomes, failure_reasons, session_end_bankrolls, remaining_cards_at_the_end_of_session, sum_total_bet_trip, sum_total_won_trip, trip_total_bets_each_bet, trip_total_won_each_bet, trip_initial_values_assignment, signaled_sides_trip = trip_result
        
        # Append trip outcomes to the overall trip results
        all_trip_results.extend(trip_outcomes)
        
        # Append session end bankrolls to the overall trip bankrolls
        all_end_of_trip_bankrolls.append(session_end_bankrolls)
        
        # Append remaining cards at the end of session to the overall remaining cards
        all_remaining_cards_at_the_end_of_session.extend(remaining_cards_at_the_end_of_session)
        
        # Append total bets and total winnings to the overall totals
        total_bet_total.extend(sum_total_bet_trip)
        total_won_total.extend(sum_total_won_trip)
        
        # Add trip total bets to the overall total bets for each bet
        for main_or_side, amount in trip_total_bets_each_bet.items():
            if main_or_side in all_total_bets_each_bet:
                all_total_bets_each_bet[main_or_side] += amount
            else:
                all_total_bets_each_bet[main_or_side] = amount
        
        # Add trip total winnings to the overall total winnings for each bet
        for main_or_side, amount in trip_total_won_each_bet.items():
            if main_or_side in all_total_won_each_bet:
                all_total_won_each_bet[main_or_side] += amount
            else:
                all_total_won_each_bet[main_or_side] = amount
        
        # Calculate the sum of initial values assignment for all trips
        all_initial_values_assignment += trip_initial_values_assignment
        
        # Calculate the total bankroll at the end of each trip
        totals_end_of_trip_bankroll.append(sum(session_end_bankrolls))
        
        # Append signaled sides for each trip to the overall signaled sides
        signaled_sides_multitrip.extend(signaled_sides_trip)

    # Calculate the average remaining cards at the end of session
    avg_remaining_cards_at_the_end_of_session = sum(all_remaining_cards_at_the_end_of_session) / len(all_remaining_cards_at_the_end_of_session)

    # Calculate the average bankrolls and standard deviations for each player
    avg_bankrolls = [statistics.mean([bankrolls[i] for bankrolls in all_end_of_trip_bankrolls]) for i in range(len(players))]
    std_bankrolls = [statistics.stdev([bankrolls[i] for bankrolls in all_end_of_trip_bankrolls]) if len([bankrolls[i] for bankrolls in all_end_of_trip_bankrolls]) > 1 else 0 for i in range(len(players))]

    # Return all the trip results
    return all_trip_results, avg_bankrolls, std_bankrolls, avg_remaining_cards_at_the_end_of_session, total_bet_total, total_won_total, all_total_bets_each_bet, all_total_won_each_bet, all_initial_values_assignment, totals_end_of_trip_bankroll, signaled_sides_multitrip


def print_final_summary(session_summary):
    # Calculate total failures and bankruptcy
    total_failures = session_summary["punter_failures"] + session_summary["counter_failures"]
    total_bankruptcy = session_summary["bankruptcies"]

    # Calculate bankruptcy percentage and completed trips ratio
    bankruptcy_percentage = total_bankruptcy / num_trips
    completed_trips_ratio = 1 - (total_bankruptcy / num_trips)

    # Calculate punter failures ratio and counter failures ratio
    punter_failures_ratio = session_summary["punter_failures"] / total_failures if total_failures != 0 else 0
    counter_failures_ratio = session_summary["counter_failures"] / total_failures if total_failures != 0 else 0

    # Print the final summary
    print("===== Final Summary =====")
    print(f"Total Sessions: {session_summary['total_sessions']}")
    print(f"Completed Trips Ratio: {completed_trips_ratio * 100:.2f}%")
    print(f"Bankruptcy Ratio: {bankruptcy_percentage * 100:.2f}%")
    print(f"Punter Failures: {session_summary['punter_failures']}")
    print(f"Counter Failures: {session_summary['counter_failures']}")
    print(f"Punter Failures Ratio: {punter_failures_ratio * 100:.2f}%")
    print(f"Counter Failures Ratio: {counter_failures_ratio * 100:.2f}%")
    print("==========================")

    # Return the calculated ratios and percentages
    return completed_trips_ratio, punter_failures_ratio, counter_failures_ratio, bankruptcy_percentage


def create_unique_filename(base_filename, directory):
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Split the base filename into name and extension
    filename, extension = os.path.splitext(base_filename)

    # Initialize the counter and the new filename
    counter = 1
    new_filename = os.path.join(directory, base_filename)

    # Check if the filename already exists, if so, append a counter to make it unique
    while os.path.exists(new_filename):
        new_filename = os.path.join(directory, f"{filename}_{counter}{extension}")
        counter += 1

    return new_filename


# Initialization and simulation
players = [Player(name="Counter_" + str(i), role="counter", counting_values=counting_values) for i in
           range(num_players - num_punters)]
players.extend([Player(name="punter_" + str(i), role="punter", counting_values=counting_values) for i in
                range(num_punters)])

# Define the punters and make sure there is at least one
punters = [player for player in players if player.role == "punter"]
counters = [player for player in players if player.role == "counter"]

assign_sides_to_players()


def run_simulation():
    with multiprocessing.Manager() as manager:
        # 1. Creation of the dictionary called session_summary 
        session_summary = manager.dict({
            "total_sessions": 0,
            "completed_sessions": 0,
            "punter_failures": 0,
            "counter_failures": 0,
            "bankruptcies": 0
        })
        
        # Check if there are punters and counters
        if not punters:
            print("THERE ARE NO PUNTERS, THE GAME CONTINUES WITH ONLY COUNTERS")

        if not counters:
            print("THERE ARE NO COUNTERS, THE GAME CONTINUES WITH ONLY PUNTERS")

        # Simulate multiple trips and get the results
        (all_trip_results, avg_bankrolls, std_bankrolls, avg_remaining_cards_at_the_end_of_session,
         money_bet_total, money_won_total, all_total_bets_each_bet,
         all_total_won_each_bet, all_initial_values_assignment, totals_end_of_trip_bankroll,
         signaled_sides_multitrip) = simulate_multiple_trips(session_summary)
        profiler.disable()
        
        # Create the stats
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        
        # Calculate total bet and total won for each bet type
        total_bet = {}
        for main_or_side, amount in all_total_bets_each_bet.items():
            total_bet[f"Total bet on {main_or_side}"] = amount
        
        total_won = {}
        for main_or_side, amount in all_total_won_each_bet.items():
            total_won[f"Total won on {main_or_side}"] = amount

        # Calculate the percentage of money won over money bet
        sum_money_bet_total = sum(s for s in money_bet_total if s is not None)
        sum_money_won_total = sum(s for s in money_won_total if s is not None)
        money_won_over_bet_perc = sum_money_won_total / sum_money_bet_total
        
        # Check the consistency of the results
        num_bets = len([s for s in money_bet_total if s is not None])
        results_count = len([s for s in all_trip_results if s is not None])

        if num_bets != results_count:
            print(f"ERROR: Number of bets ({num_bets}) different from the number of results ({results_count})")
        if sum(all_total_bets_each_bet.values()) != sum_money_bet_total:
            print("ERROR: Sum of money wagered across all bets other than total sum of money wagered per round")
        # print(f"all_total_won_each_bet: {all_total_won_each_bet}")
        if round(sum(all_total_won_each_bet.values())) != round(sum_money_won_total):
            print(
                f"ERROR: Sum of money won over all the bets ({sum(all_total_won_each_bet.values())}) different from the sum of money won per round ({sum_money_won_total})")
        if len(money_bet_total) != len(money_won_total):
            print(
                f"ERROR: Number of bets placed ({len(money_bet_total)}) different from number of paying bets({len(money_won_total)})")

        # Print the percentage of money won over money bet
        print(f"Percentage of money won over money bet: {money_won_over_bet_perc * 100:.2f}%")

        # Calculate the percentage of winning trips
        winning_trips_perc = sum(1 for total in totals_end_of_trip_bankroll if total > initial_stake) / len(
            totals_end_of_trip_bankroll)

        # Calculate the percentage of money won for each bet type
        money_won_per_bet_type_perc = {
            f"RTP on {main_or_side}": all_total_won_each_bet[main_or_side] / amount
            for main_or_side, amount in all_total_bets_each_bet.items()
            if amount > 0
        }
        for type_description, percentage in money_won_per_bet_type_perc.items():
            print(f"{type_description}: {percentage:.2f}%")

        # Calculate the average bankroll and standard deviation for each player
        player_avg_bankroll_and_std = {}
        for i, player in enumerate(players):
            player_avg_bankroll_and_std[f"{player.name} - Average Bankroll"] = avg_bankrolls[i]

        for i, player in enumerate(players):
            player_avg_bankroll_and_std[f"{player.name} - Standard Deviation"] = std_bankrolls[i]

        # Print the average bankroll and standard deviation for each player
        for player_avg_bankroll_or_std, avg_bankroll_or_std in player_avg_bankroll_and_std.items():
            print(f"{player_avg_bankroll_or_std}: {avg_bankroll_or_std:.2f}")

        # Calculate the total average bankroll
        tot_avg_bankroll = sum(s for s in avg_bankrolls if s is not None)
        print(f"Total Average Bankroll: {tot_avg_bankroll:.2f}")
        
        # Count the number of hands won by each side
        hands_result_banker = all_trip_results.count("Banker")
        hands_result_player = all_trip_results.count("Player")
        hands_result_tie = sum(
            1 for outcome in all_trip_results if outcome is not None and outcome.startswith("Tie "))

        # Calculate the number of ties outcomes per trip
        num_ties_outcomes = {side: 0 for side in list(counting_values)}
        for outcome in all_trip_results:
            for side in num_ties_outcomes:
                if side == outcome:
                    num_ties_outcomes[side] += 1

        num_ties_outcomes_per_trip = {
            f"Number of outcomes of {side} per trip": outcomes / num_trips
            for side, outcomes in num_ties_outcomes.items()
        }

        # Calculate the percentage of ties outcomes
        num_ties_percentage = {
            f"Percentage of outcome of {side}": outcomes / results_count
            for side, outcomes in num_ties_outcomes.items()
        }

        # Count the frequency of advantage signals for each side
        side_signals_count = {side: 0 for side in list(counting_values)}
        for side in signaled_sides_multitrip:
            side_signals_count[side] += 1

        # Calculate the advantage frequency per side
        advantage_frequency_per_side = {
            f"Advantage_frequency of side {side}": value / results_count
            for side, value in side_signals_count.items()
        }

        # Calculate the percentage of hands won by each side
        banker_percentage = hands_result_banker / results_count
        punter_percentage = hands_result_player / results_count
        ties_percentage = hands_result_tie / results_count

        if hands_result_tie + hands_result_player + hands_result_banker != results_count:
            print("-------ERROR---------")
            print(
                f"Number of hands played ({results_count}) different from sum of single results ({hands_result_tie}+{hands_result_banker}+{hands_result_player} = {hands_result_tie + hands_result_player + hands_result_banker})")
        print(
            f"Avg results distribution: Banker = {banker_percentage * 100:.2f}%, Punter = {punter_percentage * 100:.2f}%, Tie = {ties_percentage * 100:.2f}%")

        for side_description, percentage in num_ties_percentage.items():
            print(f"{side_description}: {percentage:.2f}%")

        for type_description, number in num_ties_outcomes_per_trip.items():
            print(f"{type_description}: {number:.2f}")

        for side_description, frequency in advantage_frequency_per_side.items():
            print(f"{side_description}: {frequency:.2f}%")

        print(f"Hands played: {results_count}")
        hands_playable = num_trips * num_sessions * hours_per_session * hands_per_hour
        print(f"Hands potentially playable: {hands_playable}")
        print(f"Percentage of trips with more money than before: {winning_trips_perc * 100:.2f}%")

        completed_trips_ratio, punter_failures_ratio, counter_failures_ratio, bankruptcy_percentage = print_final_summary(
            session_summary)

        if all_initial_values_assignment != num_trips:
            print(f"ERROR: num ({all_initial_values_assignment}) different to trip number ({num_trips})")
        elapsed_time = time.perf_counter() - start_time
        print(f"Total time of execution: {elapsed_time / 60:.2f} minutes")
        print(f"Time per hand: {elapsed_time / results_count:.5f} second")

        earning_index = ((tot_avg_bankroll-initial_stake)/initial_stake)*winning_trips_perc
        print(f"Earning index: {earning_index}")

        # Create dictionary to save the data in Excel
        excel = {"percentage of money won over money bet:": money_won_over_bet_perc}

        other_data_excel = {
            "Total Average Bankroll": tot_avg_bankroll,
            "Average Results Distribution: Banker": banker_percentage,
            "Punter": punter_percentage,
            "Tie": ties_percentage,
            "Hands Played": results_count,
            "Potentially Playable Hands": hands_playable,
            "Percentage of Trips with More Money than Before": winning_trips_perc,
            "Total Sessions": session_summary['total_sessions'],
            "Completed Sessions": session_summary['completed_sessions'],
            "Completed Trips Ratio": completed_trips_ratio,
            "Punter Failures": session_summary['punter_failures'],
            "Counter Failures": session_summary['counter_failures'],
            "Punter Failures Ratio": punter_failures_ratio,
            "Counter Failures Ratio": counter_failures_ratio,
            "Bankruptcy Ratio": bankruptcy_percentage,
            "Total Execution Time in Minutes": elapsed_time / 60,
            "Time per Hand in Seconds": elapsed_time / results_count,
            "Earning Index": earning_index,
            "": "",
            "-----SETTINGS-----": "",
            "Number of Decks": num_decks,
            "Initial Stake": initial_stake,
            "Number of Players": num_players,
            "Number of Punters": num_punters,
            "Maximum Side Counts per Player": max_side_counts_per_player,
            "Cutoff": cutoff,
            "Standard Deviation of Cutoff": std_cutoff,
            "Number of Trips": num_trips,
            "Number of Sessions per Trip": num_sessions,
            "Hours per Session": hours_per_session,
            "Hands per Hour": hands_per_hour,
            "Sizes of Possible Bets": chips_types_list,
            "Minimum Main Bet": bet_min_main,
            "Maximum Main Bet": bet_max_main,
            "Minimum Side Bet": bet_min_side,
            "Maximum Side Bet": bet_max_side,
            "Number of Exchanges Allowed per Session": exchanges_allowed_per_player_per_session,
            "Exchanges All Together?": exchanges_all_in_one,
            "Punters' Bankroll Relative to Counters": bankroll_punters_wrt_counters,
            "Counters' Base Bet Percentage": counters_base_bet_percentages,
            "Punters' Base Bet Percentage": punters_base_bet_percentages,
            "Average Size of Counters' Main Bet": average_size_main_bet,
            "Maximum Main Limit": max_limit_bet,
            "Punters' Average Base Bet Multiplier Relative to Counters": punter_base_bet_multiplier,
            "Standard Deviation of Base Bet in Percentage": base_bet_std_perc,
            "Side Multiplier Relative to Punter's Base Bet (Already Multiplied Relative to Counter)": side_multiplier_wrt_base,
            "Percentage of Punters Playing at Least One Side": punters_playing_sides_perc,
            "Percentage of Active Sides on Which Punters Who Play at Least One Side Will Bet": sides_perc_punters_bet_on,
            "Percentage of Hands Played by Counters": hands_played_by_counters_perc,
            "Percentage of Hands Played by Punters Without Advantage": hands_played_by_punters_without_advantage_perc,
            "Tie Pay Rate": tie_pay,
            "Side Size Proportionate to Bankroll?": side_size_proportioned_to_bankroll,
            "Kelly Multiplier": kelly_multiplier,
            "----------------------------": ""
        }

        excel.update(money_won_per_bet_type_perc)
        excel.update(player_avg_bankroll_and_std)
        excel.update(other_data_excel)
        excel.update(num_ties_outcomes_per_trip)
        excel.update(num_ties_percentage)
        excel.update(advantage_frequency_per_side)
        excel.update(total_bet)
        excel.update(total_won)

        # Convert the dictionary into a pandas DataFrame
        df = pd.DataFrame(list(excel.items()), columns=['Metric', 'Value'])

        # Name of the subfolder to save the results
        results_directory = "Results"
        filename = "results.xlsx"  # Base file name

        # Create a unique file name in the subfolder
        filename = create_unique_filename(filename, results_directory)

        # Now save the DataFrame to Excel in the subfolder
        try:
            df.to_excel(filename, index=False, engine='xlsxwriter')
            print(f"The data has been exported to {filename}")
        except PermissionError:
            print(
                "Error: Permission denied. Make sure the Excel file is not open in another program and try again.")

        # Print the statistics
        profiler.dump_stats('profile.stats')

    return earning_index


if __name__ == '__main__':
    # Example of function call with custom parameters
    # custom_settings = {"num_decks": 6, "initial_stake": 50000}
    # initialize_settings(custom_settings)

    # Run the simulation with custom settings
    initialize_settings()
    run_simulation()
