import matplotlib.pyplot as plt

def from_mmin_to_kmh(arr):
    return arr*60/1000

def plot_pareto(f1_fitnesses, f2_fitnesses):
    ax = plt.gca()

    ymax = round(max(f2_fitnesses))
    ymin = round(min(f2_fitnesses))

    xmax = round(max(f1_fitnesses))
    xmin = round(min(f1_fitnesses))

    ax.set_xlim([xmin-2, xmax+2])
    ax.set_ylim([ymin-2, ymax+2])

    plt.scatter(f1_fitnesses, f2_fitnesses)
    
    plt.xlabel('f1')
    plt.ylabel('f2')
    
    plt.grid()
    plt.show()


