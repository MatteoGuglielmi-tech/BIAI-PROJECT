# TOC
<!--toc:start-->
- [Authors](#authors)
- [Dual-Objective Scheduling of Rescue Vehicles to Distinguish Forest Fires via Differential Evolution and Particle Swarm Optimization Combined Algorithm <a name="introduction"></a>](#dual-objective-scheduling-of-rescue-vehicles-to-distinguish-forest-fires-via-differential-evolution-and-particle-swarm-optimization-combined-algorithm-a-nameintroductiona)
  - [Project statement](#project-statement)
  - [Experiments Data](#experiments-data)
<!--toc:end-->

# Authors
- Guglielmi Matteo (Repo Owner) : [@github](https://github.com/MatteoGuglielmi-tech)
- Davide Guidolin (Collaborator) : [@github](https://github.com/Davide-Guidolin), [@website](https://davideguidolin.com/aboutme/)

# Dual-Objective Scheduling of Rescue Vehicles to Distinguish Forest Fires via Differential Evolution and Particle Swarm Optimization Combined Algorithm <a name="introduction"></a>
This repository contains a possible implementation of a "Dual-Objective Scheduling of Rescue Vehicles to Distinguish Forest Fires via 
Differential Evolution and Particle Swarm Optimization Combined Algorithm" with reference to the [homonymous paper](https://ieeexplore.ieee.org/document/7378948).  
This project has been carried out as practical part of the course Bio-Inspired Artificial Intelligence course.  
This project caught out attention due to its inner nature of being very close to an actual emergency situation where everything needs a 
very fast decision-making procedure. This algorithm allows to do so in a manner of seconds evaluating different possible solutions.

## Project statement
This work has been carried out as application of Evolutionary and Differential algorithms to gain practical insights and make some 
experiences with bio-inspired algorithms.  
Technically, this project is devised to provide a solution (i.e. scheduling plan) to an emergency scheduling of forest fires problem.  
In the development of this work, several realistic factors have been taken into account such as :
- terrain conditions
- wind conditions (force and direction)
- number of available fire engines in the nearest fire-station
- extinguishing power of a vehicle and arrival time 
- distances between points
- temperature of a fire point. 
The formulated problem is a **complex non-linear issue** since it consists in an integer programming problem whose complexity increases
non-linearly with the number of fire points (and the possible solutions grow accordingly) and the constraints impact each other.

## Experiments Data
The data used to achieve the results are the ones used by the authors of the reference paper. Anyway, for further detail please refer to 
the section "*V. RESULTS AND ANALYSIS*" of the [project's report](https://github.com/MatteoGuglielmi-tech/BIAI-PROJECT/blob/main/Report/IEEEtran/Guglielmi_Guidolin.pdf).

## Algorithm ingredients
The flow chart followed during this work can be analyzed in the following figure:  
![MHDP Algorithm](https://github.com/MatteoGuglielmi-tech/BIAI-PROJECT/tree/main/Report/IEEEtran/Images/mhdp_algo.png?raw=true)

## Results 
![Results](https://github.com/MatteoGuglielmi-tech/BIAI-PROJECT/tree/main/Report/IEEEtran/Images/our_results_4_runs.png) ![Authors results](https://github.com/MatteoGuglielmi-tech/BIAI-PROJECT/tree/main/Report/IEEEtran/Images/authors_results.png)

