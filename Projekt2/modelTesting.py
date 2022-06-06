from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

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


# testLabels = []
# predictedLabels = []
# for i in [1,2]:#range(test_it.__len__()):
#     img, label = test_it.next()
#     testLabels.append(label)
#     predictedLabels.append(model.predict(img).argmax(axis=-1))

# print(predictedLabels)


# evaluate model
_, acc = model.evaluate(test_it, steps=len(test_it), verbose=1)
print('> %.3f' % (acc * 100.0))
