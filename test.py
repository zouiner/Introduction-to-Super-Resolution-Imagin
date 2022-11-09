#packages
exec(open('packages.py').read())

import torch, torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn


plt.rcParams["savefig.bbox"] = 'tight'
torch.manual_seed(1)


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

set_img, img, img_rescaled, padding = set_img_lr(img, parameters)

#----------------------------------------------------------------
# img = T.ToPILImage()(set_img[0].to('cpu'))
# plt.imshow(np.asarray(img), cmap = 'gray')
# plt.show()

# _, ax = plt.subplots(ncols= NImages+1)

# img = T.ToPILImage()(img.to('cpu'))
# ax[0].imshow(np.asarray(img), cmap = 'gray')
# ax[0].axis('off')
# ax[0].set_title('Original')

# for k in range(NImages):
#     img = T.ToPILImage()(set_img[k].to('cpu'))
#     ax[k+1].imshow(np.asarray(img), cmap = 'gray')
#     ax[k+1].axis('off')
#     ax[k+1].set_title('LR ' + str(k+1))

# plt.show()




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

print(bilinterpol(img[0],0.5,0.5))