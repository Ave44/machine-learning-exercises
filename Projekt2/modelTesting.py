from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

model = keras.models.load_model('Projekt2/trainedModelV1')
imgSize = 50

# create data generator
datagen = ImageDataGenerator(
    rescale=1.0/255.0, # rescaling image values
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    )


test_it = datagen.flow_from_directory('Projekt2/data/test/',
    class_mode='categorical', batch_size=140, target_size=(200, 200))


testLabels = []
testImages = []
predictedLabels = []
for i in tqdm(range(test_it.__len__())):
    img, label = test_it.next()
    for index in range(label.shape[0]):
        testLabels.append(label[index])
        testImages.append(img[index])

        # # predictedLabels.append(np.argmax(model.predict(img),axis=1))
        # predictedLabel = model.predict(img)
        # predictedLabels.append(predictedLabel)

predicted = predictedLabel = model.predict(test_it)

predicted_classes = np.argmax(predicted, axis = 1)
for i in predicted:
    print(i)
print(predicted_classes,len(predicted_classes))
predicted_true = np.argmax(testLabels,axis = 1) 



labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "/", ".", "=", "*", "-", "+", "x", "y"]

cm = confusion_matrix(predicted_true, predicted_classes) 

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

disp.plot(cmap=plt.cm.Blues)
plt.show()


# # evaluate model
# _, acc = model.evaluate(test_it, steps=len(test_it), verbose=1)
# print('> %.3f' % (acc * 100.0))
