#packages
from tkinter import N
import numpy as np
from skimage import io
from skimage.transform import rescale, resize_local_mean
from skimage.filters import correlate_sparse
from skimage.restoration import inpaint, unsupervised_wiener #ex4, ex5
from matplotlib import pyplot as plt

from scipy.signal import convolve2d as conv2 #Ex2

################################################################
# Functions
################################################################


# Ex2
def bilinear(img, px, py):
    matA = np.array([[0.5, 0.5]])
    x1 = int(np.ceil(px))
    y1 = int(np.ceil(py))
    x2 = int(np.floor(px))
    y2 = int(np.floor(py))

    xy = img.shape
    
    if x1 >= xy[0]: 
        x1 = xy[0] - 1
        if x2 >= xy[0]:
            x1 = xy[0] - 2
            x2 = xy[0] - 1
    if y1 >= xy[1]: 
        y1 = xy[1] - 1
        if y2 >= xy[1]:
            y1 = xy[1] - 2
            y2 = xy[1] - 1
    
    matB = np.array([[img[x1][y1], img[x1][y2]], [img[x2][y1], img[x2][y2]]])
    # try:
    #     matB = np.array([[img[x1][y1], img[x1][y2]], [img[x2][y1], img[x2][y2]]])
    # except:
    #     matB = np.array([[0, 0], [0, 0]])
    matC = np.array([[0.5], [0.5]])

    ans = np.dot(matA, matB)
    ans = np.dot(ans, matC)

    return ans

def set_img_lr(img, parameters):
    
    S = parameters['S']
    NImages = parameters['NImages']
    dx = parameters['dx']
    dy = parameters['dy']
    NoiseStd = parameters['NoiseStd']
    K = parameters['K']
    
    H, W = img.shape
    h, w = int(H/S), int(W/S)
    img_rescaled = conv2(img, K)
    
    #Set initial image set
    set_img = []
    for k in range(NImages):
        smallImg = np.zeros((h,w))
        for i in range(h):
            for j in range(w):
                px = i*S + dx[k] + 0.5
                py = j*S + dy[k] + 0.5
                smallImg[i][j] = bilinear(img_rescaled,px,py) + NoiseStd * np.random.rand()
        set_img.append(smallImg)
    return set_img
    # for k in range(NImages):
    #     smallImg =  resize_local_mean(img_rescaled, (h,w))
    #     for i in range(h):
    #         for j in range(w):
    #             smallImg[i][j] = smallImg[i][j] + NoiseStd * np.random.rand()
    #     set_img.append(smallImg)
    # return set_img

#Ex4
#random coordinate to simulate a motion 
def random_coor(n):
    x = []
    y = []
    h = int(np.sqrt(n))
    i = 0
    count = 0
    while(i<n):
        temp_x = int(np.random.rand()*100 % (h+ int(count/100)))
        temp_y = int(np.random.rand()*100 % (h+ int(count/100)))
        x.append(temp_x)
        y.append(temp_y)
        for j in range(len(x) - 1):
            if temp_x == x[j] and temp_y == y[j]:
                x.pop()
                y.pop()
                i -= 1
                count += 1
                break
        i += 1
    return x, y


#Ex5
# Function Gaussuian Matrix
def gaussuian_filter(kernel_size, sigma=1, muu=0):

  # Initializing value of x,y as grid of kernel size
  # in the range of kernel size

  x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),
          np.linspace(-1, 1, kernel_size))
  dst = np.sqrt(x**2+y**2)

  # lower normal part of gaussian
  normal = 1/(2.0 * np.pi * sigma**2)

  # Calculating Gaussian filter
  gauss = np.exp(-((dst-muu)**2 / (2.0 * sigma**2))) * normal
  
  return gauss

