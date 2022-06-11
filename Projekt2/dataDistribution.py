import os
import numpy as np
import matplotlib.pyplot as plt

# Lokalizacja bazy danych
folder = 'Projekt2/dataset/' # dla Windowsa
# folder = 'dataset/' # dla Linuxa

# Lokalizacja folderu do zapisania przetworzonych danych
newFolder = 'Projekt2/data/' # dla Windowsa
# folder = 'data/' # dla Linuxa

subDirectories = []
for directory in os.listdir(folder):
    subDirectories.append(directory)

filesNum = []

for i in subDirectories:
    subDirectoryPath = folder + i
    num = 0
    for file in os.listdir(subDirectoryPath):
        num += 1
    filesNum.append(num)

colors = ['#fc9890', '#f7e892', '#d4f792', '#92f7aa', '#92f7e6', '#92e1f7', '#92b7f7', '#aaa2f5', '#f5abe9', '#f5abba', '#f5d0ab', '#f5f0ab', '#e7f5ab', '#abf5b6', '#abf5df', '#abe0f5', '#bcdaf7', '#f09c99']
np.random.seed(4)
np.random.shuffle(colors)
print(colors)

plt.figure(figsize=(8,5))
plt.bar(subDirectories, filesNum, color=colors)
plt.title("Data distribution")

plt.show()