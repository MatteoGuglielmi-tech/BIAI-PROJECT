## calling method
'''
# Run PSO
    args["fig_title"] = "PSO"
    best_individual, best_fitness, final_pop = pso.run_pso(rng,num_vars=num_vars,
                                                display=display,use_log_scale=True,
                                                **args)
    print("Best PSO fitness:", best_fitness)
'''

import poplib
from webbrowser import get
from pylab import *
import plot_utils
from inspyred import ec, benchmarks
from inspyred.ec.observers import *
import matplotlib
from dataclasses import dataclass
import random
import math
import numpy as np
from numpy.random import multinomial
import utils

@dataclass
class MHDP :
    '''Class that implements the MHDP algorithm'''
    _pop_size : int
    _args : dict 

    @property
    def attributes(self):
        '''Getter function
            Return:
            ------
                _pop_size : int
                    Size of the population to be generated
                _args : dict
                    Problem data
        '''
        return self._pop_size, self._args

    # point A and B, section III
    def init_population(self) -> list:
        '''Generating the initial population Q(0)

            Params :
            -------
                self : instance class
                args : dictionary 
                    containing all problem related data

            Return:
            -------
                q0 : list
                    initial population        
        '''

        '''
        PSEUDCODE : 
        Input: Xi
        Output: Q(0)
        (1) While (i ≤ PopSize) Do
            (2) Flag = 0;
            (3) While (Flag == 0) Do
                (4) Randomly generate an integer-valued vector including N elements and satisfying (6);
                (6) Construct an individual Xi ;
                (7) If Xi is feasible, i.e., meeting (5), then
                    (8) Q(0) = Q(0) U {Xi};
                    (9) Flag = 1;
                (10) Else
                    (11) Flag = 0;
            (12) End While;
            (13) i + +;
        (14) End While;
        '''
        pop_size, args = self.attributes
        n_fire_points = args["N"]

        initial_spread_speeds = args.get('a') * np.array(args.get('temperature')) + args.get('b') * np.array(args.get('wind_force')) + args.get('c') #v_0i : array type since I assume different points may have different temperatures T
        fire_spread_speeds = initial_spread_speeds * np.array(args.get('k_s')) * np.array(args.get('k_phi')) * np.array(args.get('k_w_ms')) # v_si # k_ are just values of different areas
        args["lower_bound_points"] = 2*fire_spread_speeds/args.get('v_m') # Li > 2vsi/vm

        ui = np.array(args.get('upper_bound_points')).astype(int)
        li = np.array(args.get('lower_bound_points')).astype(int)

        q0 = []
        i = 0
        while i < pop_size :
            flag = False
            while not flag :
                candidate = []
                for j in range(n_fire_points):
                    candidate.append(random.randint(li[j], ui[j])) # generate xi within Li and Ui and compose Xi

                if np.sum(candidate) >= args.get('K') and np.sum(candidate) <= args.get('M') :
                    q0.append(candidate) #Q(0) = Q(0) ∪ {Xi};
                    flag = True
                else : 
                    flag = False
            i += 1

        return q0

    def compute_fitness(self, candidate: list[int]) -> list[tuple]:
        '''Compute fitness of a candidate (possible solution/individual)
            Params:
            ------
                candidate : list(int)
                    encoding of an individual
                args : dict
                    contains all problem related information
            Return:
            ------
                list containid tuple of the form (f1,f2) corresponind to candidate
        '''
        pop_size, args = self.attributes

        fitness = []
        fire_points_distances = np.array(args.get('distances')) # d0_i
        vehicles_speeds = np.array(args.get('vehicles_speeds')) # v0_i
        arrival_times = fire_points_distances[0][1:]/vehicles_speeds # tA_i
        initial_spread_speeds = args.get('a') * np.array(args.get('temperature')) + args.get('b') * np.array(args.get('wind_force')) + args.get('c') #v_0i : array type since I assume different points may have different temperatures T
        fire_spread_speeds = initial_spread_speeds * np.array(args.get('k_s')) * np.array(args.get('k_phi')) * np.array(args.get('k_w_ms')) # v_si # k_ are just values of different areas
        
        fire_spread_speeds = utils.from_mmin_to_kmh(fire_spread_speeds)
        v_m = utils.from_mmin_to_kmh(args.get('v_m'))
        # (fire_spread_speeds[i] * arrival_times[i])/(c * args.get('v_m') - 2*fire_spread_speeds[i])
        #             km/h                 h                   km/h                 km/h                  => h*km*h/km*h => h       

        f1 = []
        f2 = 0
        for i, c in enumerate(candidate):
            # since v_m is considered to be the same across all fire engines \sum_{m=1}^m z_0i^*v_m reduces to x_i * v_m
            if c <= 0:
                extinguishing_time = np.inf
            else:
                extinguishing_time = (fire_spread_speeds[i] * arrival_times[i])/(c * v_m - 2*fire_spread_speeds[i]) # t_Ei
           
            f1.append(extinguishing_time) # objective of minimizing the extinguishing time of fires
            f2 += c

        f1 = np.sum(f1)
        
        return (f1, f2)


    def check_constraint(self, candidate):
        '''Check whether constraints 5 and 6 are respected
            Params:
            ------
                candidate : list(int)
                    individual encoding
                args : dict
                    parameters
            Return:
            ------
                boolean type. If True constraints respected, otherwise violations.
        '''
        pop_size, args = self.attributes

        constr5 = (np.sum(candidate) >= args.get('K') and np.sum(candidate) <= args.get('M')) 
        ui = np.array(args.get('upper_bound_points')).astype(int)
        li = np.array(args.get('lower_bound_points')).astype(int)
        bools = []
        for idx, c in enumerate(candidate) :
            bools.append(c >= li[idx] and c <= ui[idx])

        constr6 = False
        if np.all(bools) :
            constr6 = True
        
        return (constr5 and constr6)

    def check_all_candidates(self, loc : list[list[int]]) -> bool :
        pop_size, args = self.attributes
        
        checklist = []
        for c in loc :
            checklist.append(self.check_constraint(c))
        return np.all(checklist)
        
    # point C section III, calculating fitness values and screening pareto solutions
    def evaluation(self, old_pop:list[list[int]], new_pop:list[list[int]], old_archive:list[list[int]]):
        '''
        PSEUDCODE : 
        Input: Q(g - 1), Q(g)
        Output: Q(g), A(g)
        (1) Calculate the fitness values of each individual in the current population Q(g) via (4), (11);
        (2) For i = 1 to PopSize
            (3) If Xi (g) is feasible, i.e., meeting (5) and (6), and Xi (g) dominates Xi(g - 1), then
                (4) Pbest (i) = Xi (g);
            (5) End If;
            (6) If Xi (g) is feasible, and Xi (g) and Xi (g - 1) are non-dominated each other, then
                (7) Randomly choose one of them as Pbest(i);
            (8) End If;
        (9) End For;
        (10) Then the new Perato solutions in Q(g) are merged into A(g - 1);
        (11) Again screen the Pareto solutions because domination relations may exist between the new Perato solutions
        and A(g - 1);
        (12) Produce A(g);
        (13) Randomly choose one of individuals in A(g) as Gbest;
        '''
        pop_size, args = self.attributes

        new_fitnesses = []
        for idx, candidate in enumerate(new_pop):
            new_fitnesses.append((idx, self.compute_fitness(candidate))) #list of all candidates fitnesses : [(1, (f11, f21)), (2, (f12, f21)), ... , (N, (f1N, f2N))]
        
        old_fitnesses = []
        for idx, candidate in enumerate(old_pop):
            old_fitnesses.append((idx, self.compute_fitness(candidate)))

        pbests = []
        for i in range(pop_size):
            ## dominance here is defined based on objective values
            ## Solutions are selected by comparing their every objective to ensure that they are Pareto optimal solutions.
            if self.check_constraint(new_pop[i]) and new_fitnesses[i][1] < old_fitnesses[i][1]:
                pbests.append(new_pop[i])
                old_archive.pop(i) ## popping dominated solution because they have lower fitness values than individual in the current pop
                old_archive.insert(i, new_pop[i]) ## inserting new individual in the archive of best solutions

            elif self.check_constraint(new_pop[i]) and new_fitnesses[i][1] == old_fitnesses[i][1] :
                rand = random.randint(0,1)
                if rand:
                    pbests.append(new_pop[i])
                else:
                    pbests.append(old_pop[i])
            else :
                pbests.append(old_pop[i])

        # for i in range(pop_size):
        #     ## dominance here is defined based on objective values
        #     ## Solutions are selected by comparing their every objective to ensure that they are Pareto optimal solutions.
        #     if self.check_constraint(new_pop[i]) and new_fitnesses[i][1] > old_fitnesses[i][1]:
        #         pbests.append(new_pop[i])
        #         old_archive.pop(i) ## popping dominated solution because they have lower fitness values than individual in the current pop
        #         old_archive.insert(i, new_pop[i]) ## inserting new individual in the archive of best solutions

        #     elif self.check_constraint(new_pop[i]) and new_fitnesses[i][1] == old_fitnesses[i][1] :
        #         rand = random.randint(0,1)
        #         if rand:
        #             pbests.append(new_pop[i])
        #         else:
        #             pbests.append(old_pop[i])
        #     else :
        #         pbests.append(old_pop[i])
  
        rand = random.randint(0, pop_size-1)
        ## rather than defining a new archive, the old archive is updated with dominant solutions
        gbest = old_archive[rand]

        return pbests, gbest
        
    def mutation_adjustment(self, pop:list[list[int]]) -> list[list[int]]:
        '''
        PSEUDCODE : 
        Input: Q(g)
        Output: Q(g)
        (1) While (i ≤PopSize)Do
        (2)     For j =1 to N
        (3)         If (xij(g)<Li)Then
        (4)             xij(g)=Li;
        (5)         Else if (xij(g)>Ui)Then
        (6)             xij(g)=Ui;
        (7)         End If;
        (8)     End for;
        (9) End While;
        '''
        pop_size, args = self.attributes
        
        adjusted_pop = []
        
        ui = np.array(args.get('upper_bound_points')).astype(int)
        li = np.array(args.get('lower_bound_points')).astype(int)
        
        for individual in pop:
            for i, gene in enumerate(individual):
                if gene < li[i]:
                    individual[i] = li[i]
                if gene > ui[i]:
                    individual[i] = ui[i]
            
            adjusted_pop.append(individual)
        
        return adjusted_pop
    
    def mutation(self, pop:list[list[int]], pbests:list[list[int]], gbest:list[int]) -> list[list[int]]:
        '''
        Xi(g+1) = Xi(g+1) + Φ[r1(Gbest - Xi(g)) + r2(Pbest - Xi(g)) + F(Xj(g) - Xk(g))]
        '''
        pop_size, args = self.attributes
        
        mutated_pop = []

        r1 = args.get('r1')
        r2 = args.get('r2')
        f = args.get('F')

        if not 0 < f < 2:
            raise ValueError('the DE scaling factor should be in range (0, 2)')

        for i, individual in enumerate(pop):
            while True:
                j = random.randint(0, len(pop)-1)
                k = random.randint(0, len(pop)-1)

                if j != k:
                    break
        
            xj = pop[j]
            xk = pop[k]

            pbest = pbests[i]

            mutated_individual = individual
            for t, gene in enumerate(individual):
                mutated_individual[t] = gene + round(r1 * (gbest[t] - gene) + r2 * (pbest[t] - gene) + f * (xj[t] - xk[t]))

            mutated_pop.append(mutated_individual)

        mutated_pop = self.mutation_adjustment(mutated_pop)
        
        return mutated_pop
    
    def crossover(self, pop:list[list[int]]):
        '''
        PSEUDCODE : 
        Input: Q(g)
        Output: Q(g)
        (1) While (i ≤PopSize)Do
        (2)     For j =1 to N
        (3)         Randomly select neighboring individual xk(g), k  ∈ {1, 2, . . . , PopSize} and k !=i;
        (4)         If (r3 < Pc && xij(g)!=xkj(g)) Then
        (5)             xij(g)=xkj(g);
        (6)         End If;
        (7)     End for;
        (8) End While;
        '''
        pop_size, args = self.attributes
        
        crossed_pop = []

        pc = args.get('Pc')

        for individual in pop:
            for i in range(len(individual)):
                while True:
                    k = random.randint(0, len(pop)-1)

                    if k!=i:
                        break
                    
                xk = pop[k]

                r3 = random.random()

                if r3 < pc and individual[i] != xk[i]:
                    individual[i] = xk[i]
            
            crossed_pop.append(individual)
        
        return crossed_pop

    def filter_feasible(self, pop):
        feasible_sol = []
        for sol in pop:
            if self.check_constraint(sol):
                feasible_sol.append(sol)
        
        return feasible_sol

    def run_mhdp(self):
        pop_size, args = self.attributes
        
        print("[*] Generating initial population")
        pop = self.init_population()
        
        print("[*] Evaluating initial population")
        pbests, gbest = self.evaluation(pop, pop, pop)
        
        print("[*] Mutation - Crossover loop")
        for i in range(args["gmax"]):
            print(f"{i}/{args['gmax']}\r", end='')

            # mutation
            mutated_pop = self.mutation(pop, pbests, gbest)
            pbests, gbest = self.evaluation(pop, mutated_pop, pbests)

            # crossover
            crossed_pop = self.crossover(mutated_pop)
            pbests, gbest = self.evaluation(mutated_pop, crossed_pop, pbests)
            
            pop = crossed_pop
        
        return self.filter_feasible(pbests)

        
                







            








