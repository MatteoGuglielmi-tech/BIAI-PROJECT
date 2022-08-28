import matplotlib.pyplot as plt

def from_mmin_to_kmh(arr):
    return arr*60/1000

def plot_pareto(f1_fitnesses, f2_fitnesses, annotate=False):

    markers = ["o", "s", "*", "x", "v", "^", "<", ">", "p", "P"]

    ax = plt.gca()

    ymax = 0
    ymin = 100
    xmax = 0
    xmin = 100

    for f2 in f2_fitnesses:
        ymax = round(max(max(f2), ymax))
        ymin = round(min(min(f2), ymin))
    
    for f1 in f1_fitnesses:
        xmax = round(max(max(f1), xmax))
        xmin = round(min(min(f1), xmin))

    ax.set_xlim([xmin-2, xmax+2])
    ax.set_ylim([ymin-2, ymax+2])

    for i, (f1, f2) in enumerate(zip(f1_fitnesses, f2_fitnesses)):
        plt.scatter(f1, f2, label=f"run {i+1}", marker=markers[i%len(markers)])

        if annotate:
            for i in range(len(f1)):
                plt.annotate(f" ({f1[i]:.2f}, {f2[i]})", (f1[i], f2[i]))
    
    plt.legend()
    plt.xlabel('f1')
    plt.ylabel('f2')
    
    plt.grid()
    plt.show()


