import numpy as np
import pyswarms as ps
import math
from pyswarms.utils.functions import single_obj as fx

print("podpunkty a i b")
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

x_max = [2, 2]
x_min = [1, 1]
my_bounds = (x_min, x_max)

optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=my_bounds)
optimizer.optimize(fx.sphere, iters=1000)

print("\npodpunkt c")


def endurance(point):
    return math.exp(-2*(point[1]-math.sin(point[0]))**2)+math.sin(point[2]*point[3])+math.cos(point[4]*point[5])


def f(swarm):
    n_particles = swarm.shape[0]
    j = [endurance(swarm[i]) for i in range(n_particles)]
    return np.array(j)


x_max = np.ones(6)
x_min = np.zeros(6)
my_bounds = (x_min, x_max)

optimizer2 = ps.single.GlobalBestPSO(n_particles=10, dimensions=6, options=options, bounds=my_bounds)
optimizer2.optimize(f, iters=1000)
