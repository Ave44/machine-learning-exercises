from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cv2
from random import randint

model = keras.models.load_model('Projekt2/modelFourBlocks/trainedModelFourBlocks')

# Lokalizacja bazy danych
testDataFolder = 'Projekt2/data/test/' # dla Windowsa
# folder = 'dataset/' # dla Linuxa

imgSize = 46
testLabels = []
testImages = []

index = 0
for directory in tqdm(os.listdir(testDataFolder)):
    label = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    label[index] = 1
    subDirectoryPath = testDataFolder + directory
    for file in os.listdir(subDirectoryPath):
        filePath = subDirectoryPath + "/" + file
        img = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
        testLabels.append(label)
        testImages.append(img)
    index += 1


# Normalizacja obrazów
datagen = ImageDataGenerator(rescale=1.0/255.0)

test_it = datagen.flow_from_directory('Projekt2/data/test/',
    class_mode='categorical', batch_size=140, target_size=(imgSize, imgSize), color_mode='grayscale', shuffle=False)

predicted = model.predict(test_it)

# Macierz błędów
predicted_classes = np.argmax(predicted, axis = 1)
predicted_true = np.argmax(testLabels, axis = 1)

labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "/", ".", "=", "*", "-", "+", "x", "y"]

cm = confusion_matrix(predicted_true, predicted_classes) 

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

disp.plot(cmap=plt.cm.Blues)

# Wyświetlanie błędnie przydzielonych obrazów
errors = []
for i in range(len(predicted_classes)):
    if predicted_classes[i] != predicted_true[i]:
        errors.append([testImages[i],i])

plt.figure()
def sub(index, n):
    pred = labels[predicted_classes[errors[n][1]]]
    true = labels[predicted_true[errors[n][1]]]
    plt.subplot(4,4,index)
    plt.imshow(errors[n][0], cmap='gray')
    plt.title(f"Pred {pred} True {true}", fontdict={'fontsize': 10})
    plt.axis('off')

for i in range(16):
    rand = randint(0, len(errors)-1)
    sub(i+1,rand)

plt.show()

# Ocena modelu (94.949%)
_, acc = model.evaluate(test_it, steps=len(test_it), verbose=1)
print('Accuracy: %.3f' % (acc * 100.0))
