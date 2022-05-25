from matplotlib import pyplot as plt
from matplotlib.image import imread
import numpy as np
from tqdm import tqdm # wyświetla pasek postępów
import cv2 # praca z obrazami

# Lokalizacja bazy danych
folder = 'Lab09/dogs-cats-mini/' # dla Windowsa
# folder = 'dogs-cats-mini/' # dla Linuxa

imagesAmount = 1250
imgSize = 50

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



# Wczytywanie obrazów psów i któw zapisywanie ich do tabel
dogsData = np.empty(imagesAmount, dtype=object)
catsData = np.empty(imagesAmount, dtype=object)

for i in tqdm(range(imagesAmount), desc='Loading dog images'):
	img = cv2.imread(f"{folder}dog/dog.{i}.jpg", cv2.IMREAD_GRAYSCALE)
	img = cv2.resize(img, (imgSize,imgSize))
	dogsData[i] = [img/255, [1, 0]] # [1, 0] to label'a oznaczająca psa

for i in tqdm(range(imagesAmount), desc='Loading cat images'):
	img = cv2.imread(f"{folder}cat/cat.{i}.jpg", cv2.IMREAD_GRAYSCALE)
	img = cv2.resize(img, (imgSize,imgSize))
	catsData[i] = [img/255, [0, 1]] # [1, 0] to label'a oznaczająca kota


# wyświetlenie obrazów po obróbce
plt.figure()
for i in range(9):
	plt.subplot(330 + 1 + i)
	plt.imshow(dogsData[i][0], cmap='gray')

plt.figure()
for i in range(9):
	plt.subplot(330 + 1 + i)
	plt.imshow(catsData[i][0], cmap='gray')
	
plt.show()



# Podział danych na pseudo-losowe zbiory testowe i treningowe
np.random.seed(1)
np.random.shuffle(dogsData)
np.random.shuffle(catsData)
print(dogsData[2])




# from keras.models import Sequential
# from keras.layers import Conv2D
# from keras.layers import MaxPooling2D
# from keras.layers import Dense
# from keras.layers import Flatten
# from keras.optimizers import SGD

# # define cnn model
# def define_model():
# 	model = Sequential()
# 	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(50, 50, 1)))
# 	model.add(MaxPooling2D((2, 2)))
# 	model.add(Flatten())
# 	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
# 	model.add(Dense(1, activation='sigmoid'))
# 	# compile model
# 	opt = SGD(lr=0.001, momentum=0.9)
# 	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
# 	return model

# model = define_model()


# # fit model
# history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
# 	validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)


# # evaluate model
# _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
# print('> %.3f' % (acc * 100.0))
# https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols
# https://www.kaggle.com/code/yassineghouzam/introduction-to-cnn-keras-0-997-top-6