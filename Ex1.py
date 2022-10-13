#packages
import numpy as np
from skimage import io
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
K = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16

# Importing an HR ground-truth image, and simulating the LR observations
#----------------------------------------------------------------
img = io.imread('Dataset/image.jpg', as_gray=True)
io.imshow(img)
plt.show()
