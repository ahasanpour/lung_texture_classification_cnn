
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,LeakyReLU
from keras.layers import Conv2D,AveragePooling2D, MaxPooling2D, ZeroPadding2D

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import numpy


def create_model():
    model = Sequential()
    #model.add(ZeroPadding2D((1,1),input_shape=input_shape))
    model.add(Conv2D(16, kernel_size=(2, 2), activation='relu', input_shape=input_shape))
    model.add(AveragePooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.5))
    model.add(Conv2D(36, (2, 2), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2,2)))

    model.add(Dropout(0.5))
    model.add(Conv2D(64, (2,2), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    #
    model.add(Dropout(0.5))
    model.add(Conv2D(100, (2,2), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))

    # model.add(Dropout(0.5))
    # model.add(Conv2D(144, (2,2), activation='relu'))
    # model.add(AveragePooling2D(pool_size=(2, 2)))




    model.add(Flatten())
    model.add(Dense(864, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(288, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model


def train_and_evaluate_model(model):
    filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode=max)
    callbacks_list = [checkpoint]

    #-------------------------------------------------
    train_datagen = ImageDataGenerator(
        rotation_range=90,
        rescale=1. / 255,
        #shuffle=True,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        #shear_range=0.2,
        #zoom_range=0.2,
        horizontal_flip=True,
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        '/home/ahp/MyMatlabProjects/lung/datasetILD/image/train',
        target_size=(32, 32),
        batch_size=16,
        shuffle=True,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        '/home/ahp/MyMatlabProjects/lung/datasetILD/image/validation',
        target_size=(32, 32),
        batch_size=16,
        class_mode='categorical')

    model.fit_generator(
        train_generator,
        steps_per_epoch=12000,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=2000,verbose=1,callbacks=callbacks_list)
    class_dictionary = validation_generator.class_indices


    target_names = ['class 0', 'class 1', 'class 2']

    return model.model.history.history['val_acc']


def model_test():
    global batch_size
    batch_size = 16
    global num_classes
    num_classes = 5
    global epochs
    epochs = 10
    global img_rows, img_cols
    img_rows, img_cols = 32, 32

    n_folds = 1


    global input_shape
    input_shape = (img_rows, img_cols,3)

    results=[]

    model = create_model()

    results.append(train_and_evaluate_model(model))
    print (results)

    print(numpy.sum(results)/n_folds)
if __name__ == '__main__':
    model_test()