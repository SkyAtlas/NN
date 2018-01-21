
# coding: utf-8

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense
from keras.utils import np_utils

# hyperparameters
batch_size = 128
num_epoch = 30
hidden_size = 512

num_train = 60000
num_test = 10000

width, height, depth = 28, 28, 1
num_classes = 10

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(num_train, width*height*depth)
X_test = X_test.reshape(num_test, width*height*depth)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

inp = Input(shape = (height*width*depth,))
hidden_1 = Dense(hidden_size, activation ='relu')(inp)
hidden_2 = Dense(hidden_size, activation = 'relu')(hidden_1)
hidden_3 = Dense(hidden_size, activation = 'relu')(hidden_2)
out = Dense(num_classes, activation = 'softmax')(hidden_3)

model = Model(input=inp, output=out)

model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

model.fit(X_train, Y_train,
          batch_size = batch_size, nb_epoch = num_epoch,
          verbose = 1, validation_split =  0.1)

model.evaluate(X_test, Y_test, verbose=1)
