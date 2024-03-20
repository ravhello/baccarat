# UR Way Egalite Baccarat Analysis

## Overview
This project presents a Python script for analyzing the vulnerability of the "UR Way Egalite" (UWE) side bet in baccarat to card counting techniques. It highlights the potential for profit through both individual and coordinated team efforts, based on detailed statistical analysis and simulations.
It includes a simulation module, an optimizer for simulation parameters, and a player class to model individual bettors' behavior and strategy.

## Components
- `baccarat_simulation_modular.py`: Runs the simulation, leveraging statistical libraries for complex game scenarios.
- `parameters_optimizer.py`: Uses machine learning techniques (specifically Gaussian process minimization) to find optimal betting strategies within the simulation environment.
- `player_class.py`: Models a player's decisions, tracking their bankroll, bet choices, and the effectiveness of card counting strategies.

## How It Works
The script simulates various betting scenarios on the UWE side bet, calculating the house edge and the effectiveness of different card counting systems for each of the ten possible tie bets. It provides a detailed analysis of the edge achievable under optimal conditions, demonstrating how players can gain a significant advantage.

## How to Use
### Prerequisites
- Python 3.x
- pandas
- NumPy
- scikit-optimize (`skopt`)

### Running the Simulation
1. Ensure all required Python packages are installed.
2. Download the script to your local machine.
3. Open your command line or terminal.
4. Navigate to the directory containing the script.
3. Use the following command to start the simulation with default parameters:
   ```bash
   python3 baccarat_simulation_modular.py

### Running the optimization
To optimize parameters for the simulation, run:
```bash
python3 parameters_optimizer.py
```
## Disclaimer
This project is for educational purposes only. It aims to explore mathematical strategies in gambling and should not be used for actual betting or any form of gambling.

## Contributing
Contributions to the project are welcome. Please ensure to follow the project's coding standards and submit pull requests for any enhancements.
