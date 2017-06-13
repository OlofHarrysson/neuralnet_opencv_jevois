import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # Removes warning printouts
import numpy as np
import cv2
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

def bin_ndarray(ndarray, new_shape, operation='mean'):
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray



model = load_model('data/models/conv_1epoch_expanded.h5')


pred_vals = []

# cap = cv2.VideoCapture(3) # Camera raw
cap = cv2.VideoCapture(0) # CamTwist
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read() # ret is boolean
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    image_out = gray[0:160, 220:420] # Cuts out the region of interest
    image = cv2.bitwise_not(image_out) # Invert image to make it look like mnist

    img = image[:, 20:180]
    img = bin_ndarray(img, (32,32))
    img = img[2:30, 2:30]
    min_brightness = np.amin(img)
    tolerance = 40
    img[img < min_brightness + tolerance] = 0

    img_input = np.expand_dims(img, 0)
    img_input = np.expand_dims(img_input, 0)

    pred = model.predict(img_input, batch_size=1, verbose=0)
    pred_vals.append(np.argmax(pred))

    if len(pred_vals) == 10:
        counts, indices = np.unique(pred_vals, return_inverse=True)
        pred_max = counts[np.argmax(np.bincount(indices))]
        print(pred_max)
        pred_vals = []


    # Display the resulting frame
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        np.save('img/test', image)
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

cv2.waitKey(1)


