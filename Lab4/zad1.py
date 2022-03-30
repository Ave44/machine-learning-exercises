import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt

# c1   - współczynnik kognitywny określa jak bardzo cząstka dąży do -Swojego- najlepszego rozwiązania
# c2   - współczynnik kognitywny określa jak bardzo cząstka dąży do -Lokalnego- najlepszego rozwiązania
# w    - współczynnik bezwładności określa jak duży wpływ ma "stara" prędkość na "nową"
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

# Call instance of GlobalBestPSO
optimizer = ps.single.GlobalBestPSO(n_particles=10,
                                    dimensions=2,
                                    options=options)

# Perform optimization
stats = optimizer.optimize(fx.sphere, iters=100)

print("-----")
print(stats)
# print(optimizer.cost_history)
# print(optimizer.pos_history)


# wyświetlanie wykresu
plot_cost_history(cost_history=optimizer.cost_history)
plt.show()