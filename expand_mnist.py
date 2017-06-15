import numpy as np
import sys
from keras.datasets import mnist
import matplotlib.pyplot as plt
import pickle

def expand_img(img):
    img_up = img[4:, :]
    img_up = np.append(img_up, np.zeros((4,28)), axis=0)
    img_left = img[:, 4:]
    img_left = np.append(img_left, np.zeros((28,4)), axis=1)
    img_down = img[0:-4, :]
    img_down = np.insert(img_down, obj=0, values=np.zeros((4,28)), axis=0)
    img_right = img[:, 0:-4]
    img_right = np.insert(img_right, obj=0, values=np.zeros((4,28)), axis=1)

    # Uncomment to plot images
    plt.subplot(222)
    plt.imshow(img_up, cmap=plt.get_cmap('gray'))
    plt.subplot(223)
    plt.imshow(img_down, cmap=plt.get_cmap('gray'))
    plt.subplot(224)
    plt.imshow(img_left, cmap=plt.get_cmap('gray'))
    plt.subplot(221)
    plt.imshow(img_right, cmap=plt.get_cmap('gray'))
    plt.show()
    sys.exit(1)

    return (img, img_up, img_left, img_down, img_right)


(X_train, y_train), (X_test, y_test) = mnist.load_data()


expanded_train = []
for image, label in zip(X_train, y_train):
    exp_img = expand_img(image)
    for img in exp_img:
        expanded_train.append((img, label))


expanded_test = []
for image, label in zip(X_train, y_test):
    exp_img = expand_img(image)
    for img in exp_img:
        expanded_test.append((img, label))


np.random.shuffle(expanded_train)
np.random.shuffle(expanded_test)

train_x = []
train_y = []
for img, label in expanded_train:
    train_x.append(img)
    train_y.append(label)


test_x = []
test_y = []
for img, label in expanded_test:
    test_x.append(img)
    test_y.append(label)


train_x = np.asarray(train_x)
test_x = np.asarray(test_x)

# Pickle has problems with to large files
with open('data/expanded_mnist_train', 'wb') as fp:
    pickle.dump((train_x, train_y), fp)

with open('data/expanded_mnist_test', 'wb') as fp:
    pickle.dump((test_x, test_y), fp)