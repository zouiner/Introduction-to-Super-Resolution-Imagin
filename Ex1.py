#packages
from tkinter import N
import numpy as np
from skimage import io
from skimage.transform import rescale, resize_local_mean
from skimage.filters import correlate_sparse
from matplotlib import pyplot as plt

# Set parameters for the degraded HR
#----------------------------------------------------------------

# Define the SR magnification
S = 3

# Define thenumber of LR images, and their offsets
NImages = 4
dx = np.array([0, 0, 2, 1])
dy = np.array([0, 1, 0, 2])

#Define the Std deviation of the noise
NoiseStd = 5/255

#Define the blur kernel
# K = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16
K = np.array([[1]])

# Importing an HR ground-truth image, and simulating the LR observations
#----------------------------------------------------------------
img = io.imread('Dataset/image.jpg', as_gray=True)
img = rescale(img, 0.25, anti_aliasing=False)

img_rescaled = correlate_sparse(img,K)

h, w = img_rescaled.shape

img_rescaled = resize_local_mean(img_rescaled, (round(h/S), round(w/S))) # Resize an array with the local mean / bilinear scaling.

h, w = img_rescaled.shape

for i in range(h):
    for j in range(w):
        img_rescaled[i][j] = img_rescaled[i][j] + NoiseStd * np.random.rand()

_, ax = plt.subplots(ncols=2)

ax[0].imshow(img, cmap = 'gray')
ax[0].axis('off')
ax[0].set_title('Original')

ax[1].imshow(img_rescaled, cmap = 'gray')
ax[1].axis('off')
ax[1].set_title("Blurred")

plt.show()