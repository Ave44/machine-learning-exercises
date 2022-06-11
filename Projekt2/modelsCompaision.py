import matplotlib.pyplot as plt

labels = ['Model V2', 'Czteroblokowy', 'Model V1', 'Dwublokowy', 'Jednoblokowy']
data = [97.222, 94.949, 93.081, 86.667, 45.556]

colors = ['#168a12', '#26b821', '#49e344', '#b3e344', '#e0e344']
print(colors)

plt.figure(figsize=(10,5))
plt.barh(labels,data, color=colors)
plt.grid(axis="x")
plt.xticks(ticks=[10,20,30,40,50,60,70,80,90,100])
plt.title("Models compasion")

plt.show()