import os
from skopt import gp_minimize, load
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.callbacks import CheckpointSaver, DeltaXStopper

# Importing the simulation function from it baccarat_v9_modular
from baccarat_simulation_modular import run_simulation, initialize_settings

# Parameter space definition for the fine-tuning of the simulation's parameters
# The objective here is to optimize the parameters in order to maximize the earning
space = [
    Integer(1, 10, name='num_players'),
    Integer(0, 10, name='num_punters'),
    Categorical([True, False], name='exchanges_all_in_one'),
    Real(0, 10, name='bankroll_punters_wrt_counters'),
    Real(1, 10, name='punter_base_bet_multiplier'),
    Real(1, 10, name='side_multiplier_wrt_base'),
    Real(0, 10, name='kelly_multiplier')
]


# Simulation function to optimize (definition)
@use_named_args(space)
def simulation(num_players, num_punters, exchanges_all_in_one, bankroll_punters_wrt_counters, punter_base_bet_multiplier, side_multiplier_wrt_base, kelly_multiplier):
    # Ensure that num_punters is not bigger than num_players, that wouldn't make sense
    if num_punters > num_players:
        num_punters = num_players

# These are the settings that the optimization function is giving as input to the module of the simulation
# Only these parameters will be optimized
    custom_settings = {
        "num_players": num_players,
        "num_punters": num_punters,
        "exchanges_all_in_one": exchanges_all_in_one,
        "bankroll_punters_wrt_counters": bankroll_punters_wrt_counters,
        "punter_base_bet_multiplier": punter_base_bet_multiplier,
        "side_multiplier_wrt_base": side_multiplier_wrt_base,
        "kelly_multiplier": kelly_multiplier
    }
    initialize_settings(custom_settings)
    earning_index = run_simulation()
    print(f"Simulation with parameters: {custom_settings}, earning index: {earning_index}")
    return -earning_index  # Negative because gp_minimize is minimizing the function, so maximizing the earning index


# Callback to print the best result after every iteration
def on_step(optim_result):
    print(f"Best result after {len(optim_result.x_iters)} calls: {-optim_result.fun}")


# Updating the result if restarting from a checkpoint, if not just creating the new checkpoint file
# This is to avoid losing the progresses of the optimization process if it stops at any time
# We did it because the optimization is taking long even when ran on AWS servers, and it may stop for any reason
def main():
    checkpoint_file = "./checkpoint.pkl"
    n_calls_total = 100
    n_calls_additional = 30
    convergence_threshold = 0.01

    if not os.path.exists(checkpoint_file):
        result_continued_from_checkpoint = gp_minimize(
            func=simulation,
            dimensions=space,
            n_calls=n_calls_total,
            random_state=0,
            callback=[CheckpointSaver(checkpoint_file), on_step, DeltaXStopper(convergence_threshold)]
        )
    else:
        result = load(checkpoint_file)
        result_continued_from_checkpoint = gp_minimize(
            func=simulation,
            x0=result.x_iters,
            y0=result.func_vals,
            n_calls=n_calls_additional,
            random_state=0,
            callback=[CheckpointSaver(checkpoint_file), on_step, DeltaXStopper(convergence_threshold)],
            dimensions=space
        )

    print("Best set of parameters:", result_continued_from_checkpoint.x)
    print("Best earning index:", -result_continued_from_checkpoint.fun)


if __name__ == '__main__':
    main()
