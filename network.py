import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # Removes warning printouts
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras import regularizers
K.set_image_dim_ordering('th')
import pickle


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

train_x = train_y = test_x = test_y = None
with open ('data/expanded_mnist_train', 'rb') as fp:
    train_x, train_y = pickle.load(fp)

with open ('data/expanded_mnist_test', 'rb') as fp:
    test_x, test_y = pickle.load(fp)


# reshape to be [samples][channels][width][height]
train_x = train_x.reshape(train_x.shape[0], 1, 28, 28).astype('float32')
test_x = test_x.reshape(test_x.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
train_x = train_x / 255
test_x = test_x / 255

train_y = np_utils.to_categorical(train_y)
test_y = np_utils.to_categorical(test_y)
num_classes = test_y.shape[1]

validation_x = test_x[:25000]
validation_y = test_y[:25000]

test_x = test_x[25000:]
test_y = test_y[25000:]

def baseline_model():
    # create model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu', kernel_regularizer=regularizers.l1(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l1(0.01)))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# build the model
model = baseline_model()
# Fit the model
model.fit(train_x, train_y, validation_data=(validation_x, validation_y), epochs=5, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(test_x, test_y, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

model.save('data/models/conv.h5')