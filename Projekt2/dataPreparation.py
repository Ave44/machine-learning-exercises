import os
import cv2
from tqdm import tqdm
import numpy as np

# Lokalizacja bazy danych
folder = 'Projekt2/dataset/' # dla Windowsa
# folder = 'dataset/' # dla Linuxa

# Lokalizacja folderu do zapisania przetworzonych danych
newFolder = 'Projekt2/data/' # dla Windowsa
# folder = 'data/' # dla Linuxa

trainPercentage = 0.8

subDirectories = []
for directory in os.listdir(folder):
    subDirectories.append(directory)

print(subDirectories)

for i in subDirectories:
    subDirectoryPath = folder + i

    # tworzenie docelowych folderów
    if not os.path.exists(newFolder + "train/" + i):
        os.makedirs(newFolder + "train/" + i)

    if not os.path.exists(newFolder + "test/" + i):
        os.makedirs(newFolder + "test/" + i)

    desc = ("Processing " + i + " images").ljust(21, " ")
    index = 0
    data = []
    for file in tqdm(os.listdir(subDirectoryPath), desc=desc):
        filePath = subDirectoryPath + "/" + file
        img = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
        imgResized = cv2.resize(img, [40,40])
        data.append(imgResized)
        index += 1

    # Pseudolosowe miesznie
    np.random.seed(1)
    np.random.shuffle(data)

    # Podział na "test" i "train" oraz zapis
    trainSize = int(len(data) * trainPercentage)
    testSize = len(data) - trainSize
    
    train = data[:trainSize]
    test = data[trainSize:]

    for n in tqdm(range(len(train)), desc=('Saving '+i+' train images')):
        cv2.imwrite(f"Projekt2/data/train/{i}/{i}.{n}.jpg", train[n])

    for n in tqdm(range(len(test)), desc=('Saving '+i+' test images ')):
        cv2.imwrite(f"Projekt2/data/test/{i}/{i}.{n}.jpg", test[n])
