#packages
exec(open('packages.py').read())

#function
from Ex2 import set_img_lr

#Parameters
S = 2
NImages = 4
dx = [0, 0, 1, 1]
dy = [0, 1, 0, 1]
NoiseStd = 0
K = [[1]]

parameters = {}
parameters['S'] = S
parameters['NImages'] = NImages
parameters['dx'] = dx
parameters['dy'] = dy
parameters['NoiseStd'] = NoiseStd
parameters['K'] = K

# Importing an HR ground-truth image, and simulating the LR observations
#----------------------------------------------------------------
img = io.imread('Dataset/image.jpg', as_gray=True)
img = rescale(img, 0.25, anti_aliasing=False)

# Create a Set of LR images
set_img = set_img_lr(img, parameters)

