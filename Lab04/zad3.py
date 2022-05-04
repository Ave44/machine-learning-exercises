import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt
from pyswarms.utils.plotters import plot_surface
from pyswarms.utils.plotters.formatters import Designer
from pyswarms.utils.plotters.formatters import Mesher

options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

# dla funkcji sphere
optimizer  = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)
stats = optimizer.optimize(fx.sphere, iters=100)
print("-----\n", stats, "\nShould be: (0, [0, 0])")

# dla funkcji ackley
optimizer2 = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)
stats2 = optimizer2.optimize(fx.ackley, iters=100)
print("-----\n", stats2, "\nShould be: (0, [0, 0])")

plot_cost_history(cost_history=optimizer.cost_history)
plot_cost_history(cost_history=optimizer2.cost_history)
plt.show()

# Tworzenie animacji 1
m = Mesher(func=fx.sphere)
pos_history_3d = m.compute_history_3d(optimizer.pos_history)

d = Designer(limits=[(-1,1), (-1,1), (-0.1,1)], label=['x-axis', 'y-axis', 'z-axis'])

animation3d = plot_surface(pos_history=pos_history_3d, 
                           mesher=m, designer=d,       
                           mark=(0,0,0))
animation3d.save('zad3-sphere.gif')



# Tworzenie animacji 2
m2 = Mesher(func=fx.ackley)
pos_history_3d2 = m2.compute_history_3d(optimizer2.pos_history)

d2 = Designer(limits=[(-3,3), (-3,3), (-0.1,5)], label=['x-axis', 'y-axis', 'z-axis'])

animation3d2 = plot_surface(pos_history=pos_history_3d2, 
                           mesher=m2, designer=d2,       
                           mark=(0,0,0))
animation3d2.save('zad3-ackley.gif')
