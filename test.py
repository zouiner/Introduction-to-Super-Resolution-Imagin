#packages
exec(open('packages.py').read())

import torch, torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn


plt.rcParams["savefig.bbox"] = 'tight'
torch.manual_seed(1)


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

    
    m = nn.Conv2d(1 , 1, K.shape[3], 1, 1, bias=False)
    m.weight = K
    img_rescaled = m(img/255)
    padding = max(max(dx,dy)) * 2
    img_rescaled = T.Pad(padding=padding)(img_rescaled)

    #Set initial image set
    set_img = []
    for k in range(NImages):
        smallImg = np.zeros((h,w))
        for i in range(h):
            for j in range(w):
                px = i*S + dx[k] + 0.5 + 1
                py = j*S + dy[k] + 0.5 + 1
                smallImg[i][j] = bilinear(img_rescaled[0].detach().numpy(),px,py) + NoiseStd * np.random.rand()
        set_img.append(torch.Tensor(smallImg)) #.to(torch.uint8))

    return set_img, img, img_rescaled



#Parameters
#----------------------------------------------------------------
S = 2
NImages = 4
dx, dy = random_coor(NImages)
print(dx, dy)

NoiseStd = 0/255
K = gkern(2.5)
# K = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]/16
K = np.array(K)
K = torch.Tensor(K)
K = K.unsqueeze(0).unsqueeze(0)
K = torch.nn.Parameter( K )
# K = torchvision.transforms.GaussianBlur(1, sigma=(1))

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

set_img, img, img_rescaled = set_img_lr(img, parameters)

#----------------------------------------------------------------
# img = T.ToPILImage()(set_img[0].to('cpu'))
# plt.imshow(np.asarray(img), cmap = 'gray')
# plt.show()

_, ax = plt.subplots(ncols= NImages+1)

img = T.ToPILImage()(img.to('cpu'))
ax[0].imshow(np.asarray(img), cmap = 'gray')
ax[0].axis('off')
ax[0].set_title('Original')

for k in range(NImages):
    img = T.ToPILImage()(set_img[k].to('cpu'))
    ax[k+1].imshow(np.asarray(img), cmap = 'gray')
    ax[k+1].axis('off')
    ax[k+1].set_title('LR ' + str(k+1))

plt.show()