#packages
import numpy as np
from skimage import io
from skimage.transform import rescale, resize, resize_local_mean
from skimage.filters import correlate_sparse
from skimage.restoration import inpaint, richardson_lucy, denoise_tv_chambolle #ex4, ex5
from matplotlib import pyplot as plt

from scipy.signal import convolve2d as conv2 #Ex2
import cv2
import math

import torch, torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn

import pytorch3d.ops
################################################################
# Functions
################################################################


# Ex2
# def bilinear(img, px, py):

#     x1 = int(np.ceil(px))
#     y1 = int(np.ceil(py))
#     x2 = int(np.floor(px))
#     y2 = int(np.floor(py))

#     matA = np.array([[x2 - px, px - x1]])
#     matB = np.array([[img[x1][y1], img[x1][y2]], [img[x2][y1], img[x2][y2]]])
#     # try:
#     #     matB = np.array([[img[x1][y1], img[x1][y2]], [img[x2][y1], img[x2][y2]]])
#     # except:
#     #     matB = np.array([[0, 0], [0, 0]])
#     matC = np.array([[y2 - py], [py - y1]])

#     ans = np.dot(matA, matB)
#     ans = np.dot(ans, matC)

#     return ans

def bilinterpol(img, a, b):

    x1 = int(np.ceil(a))
    y1 = int(np.ceil(b))
    x2 = int(np.floor(a))
    y2 = int(np.floor(b))

    pts = [[x1, y1, img[x1][y1]], [x1, y2, img[x1][y2]], [x2, y1, img[x2][y1]], [x2, y2, img[x2][y2]]]

    i = sorted(pts)
    (a1, b1, x11), (_a1, b2, x12), (a2, _b1, x21), (_a2, _b2, x22) = i
    if a1 != _a1 or a2 != _a2 or b1 != _b1 or b2 != _b2:
        print('The given points do not form a rectangle')
    if not a1 <= a <= a2 or not b1 <= b <= b2:
        print('The (a, b) coordinates are not within the rectangle')
    Y = (x11 * (a2 - a) * (b2 - b) +
            x21 * (a - a1) * (b2 - b) +
            x12 * (a2 - a) * (b - b1) +
            x22 * (a - a1) * (b - b1)
           ) / ((a2 - a1) * (b2 - b1) + 0.0)
    return Y

def set_img_lr(img, parameters):
    
    S = parameters['S']
    NImages = parameters['NImages']
    dx = parameters['dx']
    dy = parameters['dy']
    NoiseStd = parameters['NoiseStd']
    K = parameters['K']
    
    _, H, W = img.shape
    H = int(H - H%S)
    W = int(W - W%S)
    img = T.Resize(size=(H, W))(img)
    h, w = int(H/S), int(W/S)

    
    m = nn.Conv2d(1 , 1, K.shape[3], 1, 1, bias=False)
    m.weight = K
    img_rescaled = m(img/255)
    padding = max([abs(i) for i in dx+dy])
    img_rescaled = T.Pad(padding=padding)(img_rescaled)

    #Set initial image set
    set_img = []
    for k in range(NImages):
        smallImg = np.zeros((h,w))
        for i in range(h):
            for j in range(w):
                px = i*S + dx[k] + 0.5 + padding - 1
                py = j*S + dy[k] + 0.5 + padding - 1
                smallImg[i][j] = bilinterpol(img_rescaled[0].detach().numpy(),px,py) + NoiseStd * np.random.rand()
        set_img.append(torch.Tensor(smallImg)) #.to(torch.uint8))

    return set_img, img, img_rescaled, padding

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


def deconv2DTV(img, kern, lamb = 0.06, iterations = 600, epsilon = 0.4*1e-2):
    f = np.copy(img)
    tau = 1.9 / ( 1 + lamb * 8 / epsilon)
    for i in range(iterations):
        e = cv2.filter2D(f, -1, kern) - img
        Gr = np.gradient(f)
        div = sum(Gr/np.linalg.norm(Gr))
        f -= tau*(cv2.filter2D(e, -1, kern) + lamb*(div))
    return f

# Function Gaussuian Matrix
def gkern(sig, l=3):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

#Ex6 
# Function for downstamping with color images
def set_img_LR_CFA(img, parameters):
    S = parameters['S']
    NImages = parameters['NImages']
    dx = parameters['dx']
    dy = parameters['dy']
    NoiseStd = parameters['NoiseStd']
    K = parameters['K']
    CFA = parameters['CFA']

    H, W, C = img.shape
    H = int(H - H%S)
    W = int(W - W%S)
    img = resize(img, (H, W, 3))
    h, w = int(H/S), int(W/S)
    img_rescaled = cv2.filter2D(img, -1, K)

    set_img = []
    for k in range(NImages):
        smallImg = np.zeros((h,w,3))
        for i in range(h):
            for j in range(w):
                px = i*S + dx[k] + 0.5
                py = j*S + dy[k] + 0.5
                CFA_c = CFA[i%2][j%2]
                smallImg[i][j][CFA_c] = bilinear(img_rescaled[:,:,CFA_c],px,py) + NoiseStd * np.random.rand()
        set_img.append(smallImg)
    return set_img, img, img_rescaled


# Ex8
def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).to(target_type)
    return new_img