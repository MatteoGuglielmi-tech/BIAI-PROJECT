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

#def RandIntVec(ListSize : int, ListSumValue : int) -> list(int):
    #"""
    #Normal Distribution Construction....It's very flexible and hideous
    #Assume a +-3 sigma range. 
    #If one wishes to explore a different range, then changes the LowSigma and HighSigma values
    #"""
    #LowSigma    = -3#-3 sigma
    #HighSigma   = 3#+3 sigma
    #StepSize    = 1/(float(ListSize) - 1)
    #ZValues     = [(LowSigma * (1-i*StepSize) +(i*StepSize)*HighSigma) for i in range(int(ListSize))]
    ##Construction parameters for N(Mean,Variance) - Default is N(0,1)
    #NormalDistro= list()
    #for i in range(len(ZValues)):
        #if i==0:
            #ERFCVAL = 0.5 * math.erfc(-ZValues[i]/math.sqrt(2))
            #NormalDistro.append(ERFCVAL)
        #elif i ==  len(ZValues) - 1:
            #ERFCVAL = NormalDistro[0]
            #NormalDistro.append(ERFCVAL)
        #else:
            #ERFCVAL1 = 0.5 * math.erfc(-ZValues[i]/math.sqrt(2))
            #ERFCVAL2 = 0.5 * math.erfc(-ZValues[i-1]/math.sqrt(2))
            #ERFCVAL = ERFCVAL1 - ERFCVAL2
            #NormalDistro.append(ERFCVAL)  
        #Values = multinomial(ListSumValue,NormalDistro,size=1)
        #OutputValue = Values[0]
    #return OutputValue


def compute_fitness(candidate:list(int), args:dict) -> list(tuple):
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
    fitness = []
    fire_points_distances = np.array(args.get('distances')) # d0_i
    vehicles_speeds = np.array(args.get('vehicles_speeds')) # v0_i
    arrival_times = fire_points_distances/vehicles_speeds # tA_i
    initial_spread_speeds = args.get('a') * np.array(args.get('temperature')) + args.get('b') * np.array(args.get('wind_force')) + args.get('c') #v_0i : array type since I assume different points may have different temperatures T
    fire_spread_speeds = initial_spread_speeds * np.array(args.get('k_s')) * np.array(args.get('k_phi')) * np.array(args.get('k_w')) # v_si # k_ are just values of different areas
    f1 = 0
    for c in candidate:
        # since v_m is considered to be the same across all fire engines \sum_{m=1}^m z_0i^*v_m reduces to x_i * v_m
        extinguishing_times = (fire_spread_speeds * arrival_times)/(c * args.get('v_m') - 2* fire_spread_speeds) # t_Ei
        f1 += extinguishing_times # objective of minimizing the extinguishing time of fires
        f2 += c
        fitness.append((f1,f2))
    return fitness


def check_constraint(candidate, args):
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
    constr5 = (np.sum(candidate) >= args.get('K') and np.sum(candidate) <= args.get('M')) 
    ui = np.array(args.get('upper_bound_points'))
    li = np.array(args.get('lower_bound_points'))
    bools = []
    for idx, c in enumerate(candidate) :
        bools.append(c >= li[idx] and c<=ui[idx])
    if np.all(bools) :
        constr6 = True
    
    return (constr5 and constr6)

def check_all_candidates(loc : list(list(int)), args:dict) -> bool :
    checklist = []
    for c in loc :
        checklist.append(check_constraint(c, args))
    return np.all(checklist)


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


        pop_size, args = self.attributes()
        i = 0
        ui = np.array(args.get('upper_bound_points'))
        li = np.array(args.get('lower_bound_points'))
        candidate = []
        q0 = []
        while i <= pop_size :
            flag = False
            while not flag :
                candidate.append(random.randint(li[i], ui[i])) # generate xi within Li and Ui and compose Xi
                if np.sum(candidate) >= args.get('K') and np.sum(candidate) <= args.get('M') :
                    q0 += candidate #Q(0) = Q(0) ∪ {Xi};
                    flag = True
                else : 
                    flag = False
            i += 1
        return q0
        
    # point C section III, calculating fitness values and screening pareto solutions
    def evaluation(self, old_pop:list(list(int)), new_pop:list(list(int)), old_archive:list(list(int)), args:dict):
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
        new_fitnesses = []
        for idx, candidate in enumerate(new_pop):
            new_fitnesses.append((idx, compute_fitness(candidate, args))) #list of all candidates fitnesses : [(1, (f11, f21)), (2, (f12, f21)), ... , (N, (f1N, f2N))]
        
        old_fitnesses = []
        for idx, candidate in enumerate(old_pop):
            old_fitnesses.append((idx, compute_fitness(candidate, args)))

        pbest = []
        pop_size, args = self.attributes()
        for i in pop_size:
            ## dominance here is defined based on objective values
            ## Solutions are selected by comparing their every objective to ensure that they are Pareto optimal solutions.
            if check_all_candidates(new_pop, args) and new_fitnesses[i] > old_fitnesses[i]:
                pbest.append(new_pop[i])
                old_archive.pop(i) ## popping dominated solution because they have lower fitness values than individual in the current pop
                old_archive.insert(i, new_pop[i]) ## inserting new individual in the archive of best solutions

            elif check_all_candidates(new_pop, args) and new_fitnesses[i] == old_fitnesses[i] :
                rand = random.randint(0,1)
                if rand:
                    pbest.append(new_pop[i])
                else:
                    pbest.append(old_pop[i])
            else :
                pbest.append(old_pop[i])

        for i in pop_size:
            ## dominance here is defined based on objective values
            ## Solutions are selected by comparing their every objective to ensure that they are Pareto optimal solutions.
            if check_all_candidates(new_pop, args) and new_fitnesses[i] > old_fitnesses[i]:
                pbest.append(new_pop[i])
                old_archive.pop(i) ## popping dominated solution because they have lower fitness values than individual in the current pop
                old_archive.insert(i, new_pop[i]) ## inserting new individual in the archive of best solutions

            elif check_all_candidates(new_pop, args) and new_fitnesses[i] == old_fitnesses[i] :
                rand = random.randint(0,1)
                if rand:
                    pbest.append(new_pop[i])
                else:
                    pbest.append(old_pop[i])
            else :
                pbest.append(old_pop[i])
  
        rand = random.randint(pop_size)
        ## rather than defining a new archive, the old archive is updated with dominant solutions
        gbest = old_archive[rand]

        return pbest, gbest
        
    def mutation_adjustment(self, pop:list(list(int)), args:dict) -> list(list(int)):
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
        
        adjusted_pop = []
        
        ui = np.array(args.get('upper_bound_points'))
        li = np.array(args.get('lower_bound_points'))
        
        for individual in pop:
            for i, gene in enumerate(individual):
                if gene < li[i]:
                    individual[i] = li[i]
                if gene > ui[i]:
                    individual[i] = ui[i]
            
            adjusted_pop.append(individual)
        
        return adjusted_pop
    
    def mutation(self, pop:list(list(int)), pbests:list(list(int)), gbest:list(int), args:dict) -> list(list(int)):
        '''
        Xi(g+1) = Xi(g+1) + Φ[r1(Gbest - Xi(g)) + r2(Pbest - Xi(g)) + F(Xj(g) - Xk(g))]
        '''
        mutated_pop = []

        r1 = args.get('r1')
        r2 = args.get('r2')
        f = args.get('F')

        if not 0<f<2 :
            raise ValueError('the DE scaling factor should be in range (0,2)')

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
            for j, gene in enumerate(individual):
                mutated_individual[j] = gene + round(r1 * (gbest[j] - gene) + r2 * (pbest[j] - gene) + f * (xj[j] - xk[j]))

            mutated_pop.append(mutated_individual)

        mutated_pop = self.mutation_adjustment(mutated_pop)
        
        return mutated_pop
    
    def crossover(self, pop:list(list(int)), args:dict):
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

        crossed_pop = []

        pc = args.get('Pc')

        for individual in pop:
            for i in range(len(individual)):
                k = random.randint(0, len(pop)-1)
                xk = pop[k]

                r3 = random.random()

                if r3 < pc and individual[i] != xk[i]:
                    individual[i] = xk[i]
            
            crossed_pop.append(individual)
        
        return crossed_pop








        
                







            








