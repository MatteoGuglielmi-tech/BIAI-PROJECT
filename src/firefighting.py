from inspyred.ec.emo import Pareto
from pylab import *
import numpy as np
from inspyred.ec.variators import mutator
import copy

## CONSTANTS
'''
=======================================================================================
given parameters
'''
a = 0.053
b = 0.048
c = 0.275
v_m = 2.5 #m/min # it is assumed to be identical in all study cases because it depends on the motorcade configuration
M = 40 # parameter set arbitrarily
upper_bound = 10 # parameter set arbitrarily
k_s_meadow = 1
k_s_secondary = 0.7
k_s_coniferous = 0.4
'''
=======================================================================================
'''
K = 29 ## defined as sum of all L_i. Needed to be redefined as np.sum(lower_bounds)
F = 1.5
PopSize = 50
p_c = 0.6

## Concerning v_w and k_phi look at the tables IX and X in the paper


# Constraint 6 -> number of vehicles sent to fire point must be between L[i] and U[i]
class VehicleBounder(object):
    def __call__(self, candidate, args):
        low = args.get('n_vehicles_low')
        high = args.get('n_vehicles_high')
        for i, c in enumerate(candidate):
            candidate[i] = min(max(c, low[i], high[i]))
        return candidate


class ConstrainedPareto(Pareto):
    def __init__(self, values=None, violations=None, ec_maximize=True):
        Pareto.__init__(self, values)
        self.violations = violations
        self.ec_maximize=ec_maximize
    
    def __lt__(self, other):
        if self.violations is None :
            return Pareto.__lt__(self, other)
        elif len(self.values) != len(other.values):
            raise NotImplementedError
        else:
            if self.violations > other.violations :
                # if self has more violations than other
                # return true if EC is maximizing otherwise false 
                return (self.ec_maximize)
            elif other.violations > self.violations :
                # if other has more violations than self
                # return true if EC is minimizing otherwise false  
                return (not self.ec_maximize)
            elif self.violations > 0 :
                # if both equally infeasible (> 0) than cannot compare
                return False
            else :
                # only consider regular dominance if both are feasible
                not_worse = True
                strictly_better = False 
                for x, y, m in zip(self.values, other.values, self.maximize):                    
                    if m:
                        if x > y:
                            not_worse = False
                        elif y > x:
                            strictly_better = True
                    else:
                        if x < y:
                            not_worse = False
                        elif y < x:
                            strictly_better = True
            return not_worse and strictly_better


class Vehicle(object):
    def __init__(self):
        self.bounder = VehicleBounder()

    def generator(self, random, args):
        size = args.get('fire_points')
        return [random.randint(0, 100) for i in range(size)]

    def evaluator(self, candidates, args):
        ## assumed candidates to be an array x_1, x-2, ..., x_N
        fitness = []
        fire_points_distances = np.array(args.get('distances')) # d0_i
        vehicles_speeds = np.array(args.get('vehicles_speeds')) # v0_i
        arrival_times = fire_points_distances/vehicles_speeds # tA_i
        initial_spread_speeds = np.array(a * args.get('temperature') + b * args.get('wind_force') + c) #v_0i : array type since I assume different points may have different temperatures T
        fire_spread_speeds = initial_spread_speeds * args.get('k_s') * args.get('k_phi') * args.get('k_w') # v_si # k_ are just values of different areas
        f1 = 0
        for c in candidates:
            # since v_m is considered to be the same across all fire engines \sum_{m=1}^m z_0i^*v_m reduces to x_i * v_m
            extinguishing_times = (fire_spread_speeds * arrival_times)/(c * v_m - 2* fire_spread_speeds)
            f1 += extinguishing_times # objective of minimizing the extinguishing time of fires
            f2 += c
            fitness.append(ConstrainedPareto([f1, f2],
                                             self.constraint_function(c),
                                             self.maximize))
        return fitness
        
    def constraint_function(self, candidates, args):
        if not self.constrained :
            return 0
        violations = 0 

        # constraint 1
        if (np.sum(candidates) <= K and np.sum(candidates) >= M) :
            # if contraint not respected we fall into an infeasible area so we penalize linearly the solution
            violations -= np.sum(candidates)

        # constraint 2
        # U_i is said to be a given parameter
        initial_spread_speeds = np.array(a * args.get('temperature') + b * args.get('wind_force') + c) #v_0i : array type since I assume different points may have different temperatures T
        fire_spread_speeds = initial_spread_speeds * args.get('k_s') * args.get('k_phi') * args.get('k_w') # v_si # k_ are just values of different areas
        # L_i should be greater than 2v_si/v_m
        # I fixed its values
        lower_bounds = np.array(2*fire_spread_speeds/v_m)
        K = np.sum(lower_bounds)
        for c in candidates:
            if not (lower_bounds<=c and upper_bound>=c):
                violations -= c
        # constraint 3
        # this contraint is implicit in the formulation
        return violations


# to modify internal values
@mutator    
def mutation_operator(random, candidate, args):
    mut_rate = args.setdefault('mutation_rate', 0.3)
    bounder = args['_ec'].bounder
    mutant = copy.copy(candidate)
    for i, m in enumerate(mutant):
        if random.random() < mut_rate:
           # mutant[i]  # TODO write mutation function here
           pass
    mutant = bounder(mutant, args)
    return mutant


    