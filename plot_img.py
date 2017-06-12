import numpy as np
import matplotlib.pyplot as plt


# img = np.loadtxt('img.out', dtype=int)
img = np.load('img/test' + '.npy')

plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.show()