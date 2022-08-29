from ast import arg
from math import exp
import sys
import random

from pyparsing import util
from mhdp import MHDP
import utils

args = {}

# Evolution parameters
args["pop_size"] = 100   # population size
args["F"] = 1.5  # scaling factor of DE
args["Pc"] = 0.6    # crossover probability
args["gmax"] = 100  # n_iterations
args["n_runs"] = 4  # number of runs

# Vehicles parameters
args["K"] = 29  # Lower bound of the total number of vehicles required for forest fire emergency scheduling
args["M"] = 40  # Upper bound of the total number of fire engines in the fire emergency scheduling centre
args["v_m"] = 2.5 # 0.15   # extinguishing speed  2.5 m/min -> 0.15 km/h
args["vehicles_speeds"] = 54    # km/h

# Fire points parameters
args["N"] = 7   # number of fire points
args["distances"] = [
    [0, 42, 56, 63, 65, 50, 66, 45],
    [42, 0, 20, 33, 45, 35, 48, 64],
    [56, 20, 0, 46, 53, 44, 42, 58],
    [63, 33, 46, 0, 60, 55, 52, 54],
    [65, 45, 53, 60, 0, 62, 64, 56],
    [50, 35, 44, 55, 62, 0, 65, 63],
    [66, 48, 42, 52, 64, 65, 0, 63],
    [45, 64, 58, 54, 56, 66, 63, 0]
]
args["upper_bound_points"] = [10, 10, 10, 10, 10, 10, 10]

# Terrain and weather parameters
# factors related with the terrain for the spread model
args["a"] = 0.053
args["b"] = 0.048
args["c"] = 0.275
args["temperature"] = [25, 23, 22, 26, 24, 23, 22]
args["wind_force"] = [2, 1, 1, 2, 2, 2, 1]
fuel_types = ["Meadow" for _ in range(args["N"])]
slopes = [10, 2, 5, 15, 13, 8, 8]

k_s_map = {
    "Meadow": 1.0,
    "Secondary forest": 0.7,
    "Coniferous forest": 0.4
}
args["k_s"] = list(map(lambda x: k_s_map[x], fuel_types))


def k_phi_map(slope):
    if slope <= -38:
        return 0.07
    elif slope <= -33:
        return 0.13
    elif slope <= -28:
        return 0.21
    elif slope <= -23:
        return 0.32
    elif slope <= -18:
        return 0.46
    elif slope <= -13:
        return 0.63
    elif slope <= -8:
        return 0.83
    elif slope <= -3:
        return 0.90
    elif slope <= 2:
        return 1.0
    elif slope <= 7:
        return 1.20
    elif slope <= 12:
        return 1.60
    elif slope <= 17:
        return 2.10
    elif slope <= 22:
        return 2.90
    elif slope <= 27:
        return 4.10
    elif slope <= 32:
        return 6.20
    elif slope <= 37:
        return 10.10
    else:
        return 17.50


args["k_phi"] = list(map(lambda x: k_phi_map(x), slopes))
v_w = {  # wind speeds per wind force level m/s and kmh
    1: 2, 
    2: 3.6, 
    3: 5.4, 
    4: 7.4, 
    5: 9.8, 
    6: 12.3, 
    7: 14.9, 
    8: 17.7, 
    9: 20.8, 
    10: 24.2, 
    11: 27.8, 
    12: 29.8
}

args["k_w"] = list(map(lambda x: exp(0.1783 * v_w[x]), args["wind_force"]))


def single_run():
    mhdp = MHDP(args["pop_size"], args)

    best_solutions = mhdp.run_mhdp()
    print(f"Number of feasible solutions: {len(best_solutions)}")

    if len(best_solutions) == 0:
        import sys
        sys.exit(0)

    f1_fitnesses = []
    f2_fitnesses = []

    for i, s in enumerate(best_solutions):
        fitnesses = mhdp.compute_fitness(s)
        print(f"Solution {i+1}: {s}")
        print(f"\tf1: {fitnesses[0]:.2f} h")
        print(f"\tf2: {fitnesses[1]} \n")

        f1_fitnesses.append(fitnesses[0])
        f2_fitnesses.append(fitnesses[1])
    
    print("\n\n")
    
    return best_solutions, f1_fitnesses, f2_fitnesses

if __name__ == '__main__':

    if len(sys.argv) > 1 :
        random.seed(int(sys.argv[1]))
    
    runs = args["n_runs"]   

    f1_fitnesses = []
    f2_fitnesses = []

    for run in range(runs):
        print(f"=========== Run {run+1}/{runs} ===========")
        best_solutions, f1, f2 = single_run()
        f1_fitnesses.append(f1)
        f2_fitnesses.append(f2)
        # utils.plot_pareto(f1_fitnesses, f2_fitnesses)
    
    utils.plot_pareto(f1_fitnesses, f2_fitnesses)
