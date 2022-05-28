from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

model = keras.models.load_model('Lab09/trainedModel')
imgSize = 50

# create data generator
#datagen = ImageDataGenerator(rescale=1.0/255.0)
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range = 0.1, # Randomly zoom image 
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images


test_it = datagen.flow_from_directory('Lab09/preprocesedData/test/',
    class_mode='binary', batch_size=64, target_size=(200, 200))


# evaluate model
_, acc = model.evaluate(test_it, steps=len(test_it), verbose=1)
print('> %.3f' % (acc * 100.0))
