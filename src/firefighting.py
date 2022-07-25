import inspyred
import numpy as np

# Constraint 6 -> number of vehicles sent to fire point i must be between L[i] and U[i]
class VehicleBounder(object):
    def __call__(self, candidate, args):
        low = args.get('n_vehicles_low')
        high = args.get('n_vehicles_high')
        for i, c in enumerate(candidate):
            candidate[i] = min(max(c, low[i], high[i]))
        return candidate

class Vehicle(object):
    def __init__(self):
        self.bounder = VehicleBounder()

    def generator(self, random, args):
        size = args.get('fire_points')
        return [random.randint(0, 100) for i in range(size)]

    def evaluator(self, candidates, args):
        fitness = []
        fire_points_distances = np.array(args.get('distances')) # d0_i
        vehicles_speeds = np.array(args.get('vehicles_speeds')) # v0_i
        arrival_times = fire_points_distances/vehicles_speeds # tA_i
        for c in candidates:
            # TODO
            pass
        return fitness
        