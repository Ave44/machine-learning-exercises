import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt
import numpy as np

S = [1, 2, 3, 6, 10, 17, 25, 29, 30, 41, 51, 60, 70, 79, 80]

options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k':2, 'p':1}

def fitness_func(solution):
    sum1 = np.sum(solution * S)
    solution_invert = 1 - solution
    sum2 = np.sum(solution_invert * S)
    fitness = np.abs(sum1-sum2)

    return fitness

def f(swarm):
    n_particles = swarm.shape[0]
    j = [fitness_func(swarm[i]) for i in range(n_particles)]
    return np.array(j)

optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=15,
options=options)
optimizer.optimize(f, iters=30, verbose=True)
cost_history = optimizer.cost_history
plot_cost_history(cost_history)
plt.show()
