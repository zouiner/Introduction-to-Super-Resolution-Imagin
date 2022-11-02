#packages
exec(open('packages.py').read())

import torch, torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn


plt.rcParams["savefig.bbox"] = 'tight'
torch.manual_seed(1)

def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = T.ToPILImage()(img.to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def bilinear(img, px, py):

    x1 = int(np.ceil(px))
    y1 = int(np.ceil(py))
    x2 = int(np.floor(px))
    y2 = int(np.floor(py))

    matA = np.array([[x2 - px, px - x1]])
    matB = np.array([[img[x1][y1], img[x1][y2]], [img[x2][y1], img[x2][y2]]])
    # try:
    #     matB = np.array([[img[x1][y1], img[x1][y2]], [img[x2][y1], img[x2][y2]]])
    # except:
    #     matB = np.array([[0, 0], [0, 0]])
    matC = np.array([[y2 - py], [py - y1]])

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
    
    _, H, W = img.shape
    H = int(H - H%S)
    W = int(W - W%S)
    img = T.Resize(size=(H, W))(img)
    h, w = int(H/S), int(W/S)

    # padding = max(max(dx,dy)) * 2
    # img = T.Pad(padding=padding)(img)
    img_rescaled = K(img)[0]


    #Set initial image set
    # set_img = []
    # for k in range(NImages):
    #     smallImg = np.zeros((h,w))
    #     for i in range(h):
    #         for j in range(w):
    #             px = i*S + dx[k] + 0.5
    #             py = j*S + dy[k] + 0.5
    #             smallImg[i][j] = bilinear(img_rescaled,px,py) + NoiseStd * np.random.rand()
    #     set_img.append(smallImg)

    set_img = []
    smallImg = torch.zeros([h, w], dtype=torch.uint8)
    for i in range(h):
        for j in range(w):
            px = i*S + dx[0]
            py = j*S + dy[0]
            smallImg[i][j] = bilinear(img_rescaled,px,py) + NoiseStd * np.random.rand()
            # smallImg[i][j] = img_rescaled[px][py]+ NoiseStd * np.random.rand()
    set_img = torch.Tensor(smallImg)
    return set_img, img, img_rescaled, smallImg



#Parameters
#----------------------------------------------------------------
S = 2
NImages = 4
dx, dy = random_coor(NImages)
print(dx, dy)

NoiseStd = 0/255
# K = gkern(2.5)
# K = np.array(K)
# K = torch.Tensor(K)
#[[1, 2, 1], [2, 4, 2], [1, 2, 1]]/16
K = torchvision.transforms.GaussianBlur(3, sigma=(0.1, 2.5))

parameters = {}
parameters['S'] = S
parameters['NImages'] = NImages
parameters['dx'] = dx
parameters['dy'] = dy
parameters['NoiseStd'] = NoiseStd
parameters['K'] = K


# Importing an HR ground-truth image, and simulating the LR observations
#----------------------------------------------------------------
img = torchvision.io.read_image('Dataset/image.jpg')
img = T.Resize(size=300)(img)
img = T.Grayscale()(img)
img = img.unsqueeze(0)
_, _, H, W = img.shape
img = F.interpolate(img.float(), [int(H/S), int(W/S)], mode='bilinear')
img.to(torch.uint8)
img = img.squeeze(0)
# set_img, img, img_rescaled = set_img_lr(img, parameters)
# print(set_img.shape, img.shape, img_rescaled.shape)



img = T.ToPILImage()(img.to('cpu'))
plt.imshow(np.asarray(img), cmap = 'gray')
plt.show()



# set_img = [i for i in set_img]
# io.imshow_collection(set_img, cmap= 'gray')
# plt.show()