import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import h5py

PATH = '../nyu_depth_v2_labeled.mat'

# read mat file
FILE = h5py.File(PATH)
# Discovering keys
keys = list(FILE.keys())
print(keys)
img = FILE['images'][0]
print(img.shape)
# reshape
img_ = np.empty([480, 640, 3])
img_[:, :, 0] = img[0, :, :].T
img_[:, :, 1] = img[1, :, :].T
img_[:, :, 2] = img[2, :, :].T

# imshow
img__ = img_.astype('float32')
io.imshow(img__ / 255.0)
io.show()

# print(img.shape)
# plt.imshow(img_/255)
# plt.show()
# See sample image
