import skimage.io as io
import numpy as np
import h5py

PATH = '../nyu_depth_v2_labeled.mat'

# read mat file
f = h5py.File(PATH)
img = f['images'][0]
print(img.shape)
