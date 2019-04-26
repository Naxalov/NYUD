import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import h5py

# PATH = '../nyu_depth_v2_labeled.mat'
PATH = 'D:/project/ML/dataset/nyu_depth_v2_labeled.mat'
# index of image (dataset)
INDEX_F = 0
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
plt.imshow(img__/255.0)
plt.show()



# read corresponding depth

depth = FILE['depths'][INDEX_F]


# reshape for imshow
depth_ = np.empty([480, 640, 3])
depth_[:,:,0] = depth[:,:].T
depth_[:,:,1] = depth[:,:].T
depth_[:,:,2] = depth[:,:].T

plt.imshow(depth_/4.0)
plt.show()