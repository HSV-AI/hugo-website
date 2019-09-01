+++
title =  "Working with Large Datasets"
date = 2019-08-21T22:40:19-05:00
tags = []
featured_image = ""
description = ""
+++

We talked about approaches to work with large datasets when training deep neural networks.

<a href="https://colab.research.google.com/github/HSV-AI/presentations/blob/master/2019/190821_Large_Data.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<!--more-->

## Agenda

* Intro & Welcome
* Project Updates
** T-Shirts
** SBIR topics drop soon
* Deep Learning with Large Data Sets
* Planning for Fall

Ben to present next week...

Who does AI for Agriculture

Lightning Talks

Phil writing program for gradient descent

How the post from Grant Silva actualy works (maybe Ben)

September 18&19 - NASA Small Business Workshop






## Deep Learning with Large Data Sets

There are times when you will need to be able to train a neural network with a dataset that is larger than can fit into the available memory of a machine.

So far, the first time I have encountered this issue is with an RNN that takes a sequence of input data.

For example, the project that I'm working with Radar and Satellite data has the following input structure:

```
year/month/day/
    radar/
        images (720x1832 float64) ~= 10MB (average ~150-300 scans per day)
    sat/
        band/
            images (each band scanned every 5 minutes)
    rain.csv
```

Assuming that the sat images are similar to the radar images, that gives us around 10GB of data per day.

The approach with Keras is to use a Data Generator to load the data incrementally when training the model.


## References:

[ImageDataGenerator](https://medium.com/datadriveninvestor/keras-imagedatagenerator-methods-an-easy-guide-550ecd3c0a92)  \[[source](https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py#L233)\]

[Multiprocess ImageDataGenerator](https://github.com/stratospark/keras-multiprocess-image-data-generator)

[Stanford Tutorial](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)

[Loading Large Data](https://medium.com/datadriveninvestor/keras-training-on-large-datasets-3e9d9dbc09d4)

[More Loading Large Datasets](https://machinelearningmastery.com/how-to-load-large-datasets-from-directories-for-deep-learning-with-keras/)

[Keras Feature extraction on large data](https://www.pyimagesearch.com/2019/05/27/keras-feature-extraction-on-large-datasets-with-deep-learning/)

[Keras Fit vs Fit_Generator](https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/)



## Keras Example

Here's an example from the Keras codebase that uses the ImageDataGenerator. You can think of this as a different approach that takes a given set of images and increases the data size by applying transformations to create new images based off of existing data.


```
import multiprocessing

multiprocessing.cpu_count()
```




    2




```
'''
#Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

from math import ceil

batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


n_points = len(x_train)

steps_per_epoch = ceil(n_points / batch_size)


# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=(x_test, y_test),
                        workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
```

    x_train shape: (50000, 32, 32, 3)
    50000 train samples
    10000 test samples
    Using real-time data augmentation.
    Epoch 1/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 1.8888 - acc: 0.3081 - val_loss: 1.5796 - val_acc: 0.4246
    Epoch 2/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 1.5950 - acc: 0.4155 - val_loss: 1.4498 - val_acc: 0.4679
    Epoch 3/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 1.4648 - acc: 0.4669 - val_loss: 1.3281 - val_acc: 0.5237
    Epoch 4/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 1.3778 - acc: 0.5040 - val_loss: 1.2068 - val_acc: 0.5666
    Epoch 5/100
    1563/1563 [==============================] - 30s 20ms/step - loss: 1.3005 - acc: 0.5367 - val_loss: 1.1448 - val_acc: 0.5890
    Epoch 6/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 1.2436 - acc: 0.5574 - val_loss: 1.2455 - val_acc: 0.5662
    Epoch 7/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 1.1940 - acc: 0.5768 - val_loss: 1.1144 - val_acc: 0.6064
    Epoch 8/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 1.1542 - acc: 0.5921 - val_loss: 1.0732 - val_acc: 0.6178
    Epoch 9/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 1.1181 - acc: 0.6051 - val_loss: 1.1320 - val_acc: 0.6046
    Epoch 10/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 1.0832 - acc: 0.6184 - val_loss: 0.9794 - val_acc: 0.6564
    Epoch 11/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 1.0646 - acc: 0.6256 - val_loss: 0.9268 - val_acc: 0.6750
    Epoch 12/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 1.0379 - acc: 0.6340 - val_loss: 0.9803 - val_acc: 0.6574
    Epoch 13/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 1.0186 - acc: 0.6419 - val_loss: 0.8696 - val_acc: 0.6991
    Epoch 14/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.9955 - acc: 0.6520 - val_loss: 0.8852 - val_acc: 0.6900
    Epoch 15/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.9792 - acc: 0.6568 - val_loss: 0.8849 - val_acc: 0.6883
    Epoch 16/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.9651 - acc: 0.6616 - val_loss: 0.8277 - val_acc: 0.7109
    Epoch 17/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.9512 - acc: 0.6670 - val_loss: 0.8411 - val_acc: 0.7088
    Epoch 18/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.9386 - acc: 0.6735 - val_loss: 0.8132 - val_acc: 0.7181
    Epoch 19/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.9312 - acc: 0.6777 - val_loss: 0.8275 - val_acc: 0.7168
    Epoch 20/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.9189 - acc: 0.6800 - val_loss: 0.8346 - val_acc: 0.7093
    Epoch 21/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.9063 - acc: 0.6850 - val_loss: 0.7739 - val_acc: 0.7340
    Epoch 22/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.9012 - acc: 0.6880 - val_loss: 0.7881 - val_acc: 0.7363
    Epoch 23/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.8904 - acc: 0.6927 - val_loss: 0.8672 - val_acc: 0.7048
    Epoch 24/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.8865 - acc: 0.6928 - val_loss: 0.8272 - val_acc: 0.7248
    Epoch 25/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.8817 - acc: 0.6959 - val_loss: 0.8464 - val_acc: 0.7149
    Epoch 26/100
    1563/1563 [==============================] - 30s 20ms/step - loss: 0.8727 - acc: 0.6987 - val_loss: 0.8260 - val_acc: 0.7244
    Epoch 27/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.8726 - acc: 0.7000 - val_loss: 0.7757 - val_acc: 0.7381
    Epoch 28/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.8650 - acc: 0.7016 - val_loss: 0.7845 - val_acc: 0.7307
    Epoch 29/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.8572 - acc: 0.7041 - val_loss: 0.7616 - val_acc: 0.7401
    Epoch 30/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.8529 - acc: 0.7066 - val_loss: 0.7692 - val_acc: 0.7424
    Epoch 31/100
    1563/1563 [==============================] - 30s 20ms/step - loss: 0.8488 - acc: 0.7080 - val_loss: 0.7834 - val_acc: 0.7339
    Epoch 32/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.8449 - acc: 0.7096 - val_loss: 0.7400 - val_acc: 0.7485
    Epoch 33/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.8411 - acc: 0.7127 - val_loss: 0.7679 - val_acc: 0.7426
    Epoch 34/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.8354 - acc: 0.7124 - val_loss: 0.7072 - val_acc: 0.7604
    Epoch 35/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.8359 - acc: 0.7159 - val_loss: 0.7698 - val_acc: 0.7367
    Epoch 36/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.8284 - acc: 0.7185 - val_loss: 0.7209 - val_acc: 0.7565
    Epoch 37/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.8296 - acc: 0.7167 - val_loss: 0.7725 - val_acc: 0.7394
    Epoch 38/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.8280 - acc: 0.7181 - val_loss: 0.7660 - val_acc: 0.7413
    Epoch 39/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.8261 - acc: 0.7185 - val_loss: 0.7471 - val_acc: 0.7467
    Epoch 40/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.8197 - acc: 0.7214 - val_loss: 0.7493 - val_acc: 0.7456
    Epoch 41/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.8157 - acc: 0.7231 - val_loss: 0.7273 - val_acc: 0.7534
    Epoch 42/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.8127 - acc: 0.7234 - val_loss: 0.7033 - val_acc: 0.7663
    Epoch 43/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.8236 - acc: 0.7195 - val_loss: 0.7225 - val_acc: 0.7541
    Epoch 44/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.8154 - acc: 0.7222 - val_loss: 0.7475 - val_acc: 0.7433
    Epoch 45/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.8135 - acc: 0.7252 - val_loss: 0.7172 - val_acc: 0.7622
    Epoch 46/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.8079 - acc: 0.7243 - val_loss: 0.7429 - val_acc: 0.7535
    Epoch 47/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.8088 - acc: 0.7264 - val_loss: 0.7987 - val_acc: 0.7318
    Epoch 48/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.8098 - acc: 0.7266 - val_loss: 0.7725 - val_acc: 0.7437
    Epoch 49/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.8017 - acc: 0.7278 - val_loss: 0.7392 - val_acc: 0.7485
    Epoch 50/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.8020 - acc: 0.7286 - val_loss: 0.6994 - val_acc: 0.7683
    Epoch 51/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.7983 - acc: 0.7329 - val_loss: 0.7315 - val_acc: 0.7562
    Epoch 52/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.8041 - acc: 0.7303 - val_loss: 0.6834 - val_acc: 0.7699
    Epoch 53/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.8016 - acc: 0.7289 - val_loss: 0.7256 - val_acc: 0.7526
    Epoch 54/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.7971 - acc: 0.7329 - val_loss: 0.7188 - val_acc: 0.7530
    Epoch 55/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7951 - acc: 0.7304 - val_loss: 0.6851 - val_acc: 0.7671
    Epoch 56/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7999 - acc: 0.7319 - val_loss: 0.7218 - val_acc: 0.7650
    Epoch 57/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.7942 - acc: 0.7322 - val_loss: 0.6609 - val_acc: 0.7779
    Epoch 58/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.7915 - acc: 0.7351 - val_loss: 0.7780 - val_acc: 0.7411
    Epoch 59/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.7861 - acc: 0.7344 - val_loss: 0.7096 - val_acc: 0.7640
    Epoch 60/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.7917 - acc: 0.7349 - val_loss: 0.7629 - val_acc: 0.7393
    Epoch 61/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.7902 - acc: 0.7356 - val_loss: 0.7593 - val_acc: 0.7501
    Epoch 62/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.7911 - acc: 0.7340 - val_loss: 0.6896 - val_acc: 0.7677
    Epoch 63/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.7861 - acc: 0.7366 - val_loss: 0.7194 - val_acc: 0.7589
    Epoch 64/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.7971 - acc: 0.7321 - val_loss: 0.7059 - val_acc: 0.7662
    Epoch 65/100
    1563/1563 [==============================] - 31s 20ms/step - loss: 0.7895 - acc: 0.7354 - val_loss: 0.7501 - val_acc: 0.7552
    Epoch 66/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7925 - acc: 0.7352 - val_loss: 0.7366 - val_acc: 0.7697
    Epoch 67/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7888 - acc: 0.7344 - val_loss: 0.7504 - val_acc: 0.7464
    Epoch 68/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7938 - acc: 0.7363 - val_loss: 0.7282 - val_acc: 0.7511
    Epoch 69/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7865 - acc: 0.7367 - val_loss: 0.7112 - val_acc: 0.7607
    Epoch 70/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7837 - acc: 0.7381 - val_loss: 0.6503 - val_acc: 0.7866
    Epoch 71/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7817 - acc: 0.7386 - val_loss: 0.7183 - val_acc: 0.7593
    Epoch 72/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7802 - acc: 0.7376 - val_loss: 0.6615 - val_acc: 0.7807
    Epoch 73/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7868 - acc: 0.7361 - val_loss: 0.6982 - val_acc: 0.7794
    Epoch 74/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7922 - acc: 0.7359 - val_loss: 0.7361 - val_acc: 0.7658
    Epoch 75/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7832 - acc: 0.7380 - val_loss: 0.8591 - val_acc: 0.7089
    Epoch 76/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7860 - acc: 0.7374 - val_loss: 0.7909 - val_acc: 0.7330
    Epoch 77/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7827 - acc: 0.7379 - val_loss: 0.6997 - val_acc: 0.7688
    Epoch 78/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7847 - acc: 0.7370 - val_loss: 0.7417 - val_acc: 0.7557
    Epoch 79/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7887 - acc: 0.7355 - val_loss: 0.6754 - val_acc: 0.7756
    Epoch 80/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7885 - acc: 0.7364 - val_loss: 0.9642 - val_acc: 0.6993
    Epoch 81/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7867 - acc: 0.7359 - val_loss: 0.7186 - val_acc: 0.7626
    Epoch 82/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7851 - acc: 0.7365 - val_loss: 0.7174 - val_acc: 0.7607
    Epoch 83/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7873 - acc: 0.7379 - val_loss: 0.7089 - val_acc: 0.7691
    Epoch 84/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7902 - acc: 0.7382 - val_loss: 0.6812 - val_acc: 0.7754
    Epoch 85/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7945 - acc: 0.7357 - val_loss: 0.7623 - val_acc: 0.7511
    Epoch 86/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7879 - acc: 0.7381 - val_loss: 0.7171 - val_acc: 0.7626
    Epoch 87/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7837 - acc: 0.7377 - val_loss: 0.7788 - val_acc: 0.7472
    Epoch 88/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7922 - acc: 0.7391 - val_loss: 0.7794 - val_acc: 0.7401
    Epoch 89/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7923 - acc: 0.7354 - val_loss: 0.7119 - val_acc: 0.7699
    Epoch 90/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7950 - acc: 0.7369 - val_loss: 0.6720 - val_acc: 0.7783
    Epoch 91/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7894 - acc: 0.7384 - val_loss: 0.7742 - val_acc: 0.7440
    Epoch 92/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7936 - acc: 0.7369 - val_loss: 0.6867 - val_acc: 0.7800
    Epoch 93/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7938 - acc: 0.7380 - val_loss: 0.6889 - val_acc: 0.7784
    Epoch 94/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7900 - acc: 0.7376 - val_loss: 0.6880 - val_acc: 0.7713
    Epoch 95/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7957 - acc: 0.7362 - val_loss: 0.6850 - val_acc: 0.7807
    Epoch 96/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7966 - acc: 0.7398 - val_loss: 0.7368 - val_acc: 0.7574
    Epoch 97/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7932 - acc: 0.7360 - val_loss: 0.7736 - val_acc: 0.7569
    Epoch 98/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7947 - acc: 0.7384 - val_loss: 0.7938 - val_acc: 0.7397
    Epoch 99/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7981 - acc: 0.7362 - val_loss: 0.7807 - val_acc: 0.7329
    Epoch 100/100
    1563/1563 [==============================] - 30s 19ms/step - loss: 0.7956 - acc: 0.7385 - val_loss: 0.7134 - val_acc: 0.7646
    Saved trained model at /content/saved_models/keras_cifar10_trained_model.h5 
    10000/10000 [==============================] - 1s 88us/step
    Test loss: 0.7133953602313995
    Test accuracy: 0.7646

