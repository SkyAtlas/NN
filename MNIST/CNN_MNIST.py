
# coding: utf-8

from keras.datasets import mnist 
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Convolution2D, MaxPooling2D, Dropout
from keras.utils import np_utils 
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import merge
from keras.callbacks import EarlyStopping

batch_size = 128
num_epochs = 50
kernel_size = 3
pool_size = 2
conv_depth = 32
drop_prob_1 = 0.25
drop_prob_2 = 0.5
hidden_size = 128

num_train = 60000
height, width, depth = 28, 28, 1
num_classes = 10

l2_lambda = 0.0001
ens_models = 3

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], height, width, depth)
X_test = X_test.reshape(X_test.shape[0], height, width, depth)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)

inp = Input(shape=(height, width, depth)) 
inp_norm = BatchNormalization(axis=1)(inp)
outs = []

for i in range(ens_models):
    conv_1 = Convolution2D(conv_depth,
                           kernel_size,
                           kernel_size,
                           border_mode='same',
                           init='he_uniform',
                           kernel_regularizer = l2(l2_lambda),
                           activation='relu')(inp_norm)
    conv_1 = BatchNormalization(axis=1)(conv_1)

    conv_2 = Convolution2D(conv_depth, 
                           kernel_size,
                           kernel_size,
                           border_mode='same',
                           init='he_uniform',
                           kernel_regularizer = l2(l2_lambda),
                           activation='relu')(conv_1)
    conv_2 = BatchNormalization(axis=1)(conv_2)
    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
    drop_1 = Dropout(drop_prob_1)(pool_1)
    flat = Flatten()(drop_1)
    hidden = Dense(hidden_size, kernel_regularizer = l2(l2_lambda), init='he_uniform', activation='relu')(flat)
    hidden = BatchNormalization(axis=1)(hidden)
    drop = Dropout(drop_prob_2)(hidden)
    outs.append(Dense(num_classes, init='glorot_uniform', activation='softmax')(drop))

out = merge(outs, mode ='ave')
model = Model(input = inp, output = out)

model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

X_val = X_train[54000:]
Y_val = Y_train[54000:]
X_train = X_train[:54000]
Y_train = Y_train[:54000]

datagen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1)
datagen.fit(X_train)
model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=num_epochs,
                        validation_data=(X_val, Y_val),
                        verbose=1,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=5)])

model.evaluate(X_test, Y_test, verbose=1)