from matplotlib import pyplot as plt
from matplotlib.image import imread
import numpy as np
from tqdm import tqdm # wyświetla pasek postępów
import cv2 # praca z obrazami

# Lokalizacja bazy danych
folder = 'Lab09/dogs-cats-mini/' # dla Windowsa
# folder = 'dogs-cats-mini/' # dla Linuxa

imagesAmount = 12500
imgSize = 50
trainPercentage = 0.7

# wyświetlenie 9 pierwszych obrazów psów i 9 pierwszych obrazów kotów
plt.figure()
for i in range(9):
	plt.subplot(330 + 1 + i)
	image = imread(f"{folder}dog/dog.{i}.jpg")
	plt.imshow(image)

plt.figure()
for i in range(9):
	plt.subplot(330 + 1 + i)
	image = imread(f"{folder}cat/cat.{i}.jpg")
	plt.imshow(image)

# Wczytywanie i obróbka obrazów psów i któw oraz zapisywanie ich do tabel
# dogsData = np.empty(imagesAmount, dtype=object)
# catsData = np.empty(imagesAmount, dtype=object)

# for i in tqdm(range(imagesAmount), desc='Loading dog images'):
# 	img = cv2.imread(f"{folder}dog/dog.{i}.jpg", cv2.IMREAD_GRAYSCALE)
# 	img = cv2.resize(img, (imgSize,imgSize))
# 	dogsData[i] = [img/255, [1, 0]] # [1, 0] to label'a oznaczająca psa

# for i in tqdm(range(imagesAmount), desc='Loading cat images'):
# 	img = cv2.imread(f"{folder}cat/cat.{i}.jpg", cv2.IMREAD_GRAYSCALE)
# 	img = cv2.resize(img, (imgSize,imgSize))
# 	catsData[i] = [img/255, [0, 1]] # [1, 0] to label'a oznaczająca kota

dogsData = np.empty((imagesAmount, imgSize, imgSize))
catsData = np.empty((imagesAmount, imgSize, imgSize))

for i in tqdm(range(imagesAmount), desc='Loading dog images'):
	img = cv2.imread(f"{folder}dog/dog.{i}.jpg", cv2.IMREAD_GRAYSCALE)
	img = cv2.resize(img, (imgSize,imgSize))
	dogsData[i] = img

for i in tqdm(range(imagesAmount), desc='Loading cat images'):
	img = cv2.imread(f"{folder}cat/cat.{i}.jpg", cv2.IMREAD_GRAYSCALE)
	img = cv2.resize(img, (imgSize,imgSize))
	catsData[i] = img

# wyświetlenie obrazów po obróbce
plt.figure()
for i in range(9):
	plt.subplot(330 + 1 + i)
	plt.imshow(dogsData[i], cmap='gray')

plt.figure()
for i in range(9):
	plt.subplot(330 + 1 + i)
	plt.imshow(catsData[i], cmap='gray')
	
plt.show()

# Podział danych na pseudo-losowe zbiory testowe i treningowe
trainSize = int(imagesAmount * trainPercentage)

np.random.seed(1)
np.random.shuffle(dogsData)
np.random.shuffle(catsData)

dogTrain = dogsData[:trainSize]
dogTest = dogsData[trainSize:]

catTrain = catsData[:trainSize]
catTest = catsData[trainSize:]



for i in tqdm(range(len(dogTrain)), desc='Saving dog train images'):
	cv2.imwrite(f"Lab09/preprocesedData/train/dogs/dog.{i}.jpg", dogTrain[i])

for i in tqdm(range(len(dogTest)), desc='Saving dog test images '):
	cv2.imwrite(f"Lab09/preprocesedData/test/dogs/dog.{i}.jpg", dogTest[i])

for i in tqdm(range(len(catTrain)), desc='Saving cat train images'):
	cv2.imwrite(f"Lab09/preprocesedData/train/cats/cat.{i}.jpg", catTrain[i])

for i in tqdm(range(len(catTest)), desc='Saving cat test images '):
	cv2.imwrite(f"Lab09/preprocesedData/test/cats/cat.{i}.jpg", catTest[i])

