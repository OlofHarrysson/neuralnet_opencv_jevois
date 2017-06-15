import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # Removes warning printouts
from keras.utils import plot_model
from keras.models import load_model
import numpy as np

def pause():
    programPause = input("Press the <ENTER> key to continue...")


model = load_model('data/models/conv_1epoch_exp_0.25_0.5drop.h5')

for layer in model.layers:
    list_weights = layer.get_weights()
    for w in list_weights:
        max_weight = np.amax(w)
        print(max_weight)

    # pause()




