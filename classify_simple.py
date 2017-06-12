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


# model = load_model('models/simple.h5')
# print(model)
# pred = predict(self, x, batch_size=32, verbose=0)



# cap = cv2.VideoCapture(3) # Camera raw
cap = cv2.VideoCapture(0) # CamTwist
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read() # ret is boolean
    # Our operations on the frame come here
    # print(frame.shape)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    image = gray[0:160, 220:420] # Cuts out the region of interest
    print(image.shape)

    # Display the resulting frame
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()




import matplotlib.pyplot as plt
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.show()

