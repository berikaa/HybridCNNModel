import keras
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from keras.layers.normalization import BatchNormalization
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from keras.optimizers import SGD
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
NAME = "lung_CNN"
train_filenames = train_filenames.astype('float32')
val_filenames = val_filenames.astype('float32')
#test_filenames = test_filenames.astype('float32')

train_filenames /= 255
val_filenames /= 255
#test_filenames /= 255

print(train_filenames.shape[0], 'train samples(eğitim örnek sayısı)')
print(val_filenames.shape[0], 'test samples(test örnek sayısı)')
#print(test_filenames.shape[0], 'test (test sayısı)')

model = Sequential()

model.add(Conv2D(128, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))

model.add(Conv2D(96, (2, 2)))
model.add(Activation('relu'))

model.add(Conv2D(64, (1, 1)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1)))

model.add(Dropout(0.5))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))

model.add(Conv2D(96, (2, 2)))
model.add(Activation('relu'))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))

model.add(AveragePooling2D(pool_size=(1, 1), strides=(1,1)))

model.add(Dropout(0.5))

model.add(Conv2D(32, (2, 2), strides=(2,2)))
model.add(Activation('relu'))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Dropout(0.5))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/fit/".format(NAME))

model.compile(loss='mean_squared_error',optimizer='rmsprop',metrics=['accuracy'])

model.summary()
#(learning_rate=0.001, rho=0.9) #...
model.fit(train_filenames, train_labels,
          batch_size=32,
          epochs=10,
          verbose=1,
          validation_data=(val_filenames, val_labels),
          callbacks=[tensorboard])
