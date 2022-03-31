import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pyswarms.utils.plotters import plot_contour
from pyswarms.utils.plotters.formatters import Mesher

# Uruchomienie optymalizacji
options = {'c1':0.5, 'c2':0.3, 'w':0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)

# historia kosztów i pozycji
cost_history, pos_history = optimizer.optimize(fx.sphere, iters=100)

# tworzenie 2 różnych animacji bez imagemacic

animation = plot_contour(pos_history=optimizer.pos_history,
                         mesher=Mesher(func=fx.sphere),
                         mark=(0,0))
animation.save('zad2.gif')


points = optimizer.pos_history

fig, ax, = plt.subplots()
ln, = plt.plot([], [], 'ro')

def init():
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    xdata = []
    ydata = []

    for i in points[frame]:
        xdata.append(i[0])
        ydata.append(i[1])

    ln.set_data(xdata, ydata)
    return ln

ani = FuncAnimation(fig, update, frames=100, init_func=init, repeat=False)
plt.show()