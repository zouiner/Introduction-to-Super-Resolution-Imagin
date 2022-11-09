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
# dx, dy = random_coor(NImages)
dx = [1, 2, 4, 3]
dy = [1, 2, 3, -2]
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

NRows = math.comb(NImages, 2)
TX = np.zeros((NRows,1))
TY = np.zeros((NRows,1))
A = np.zeros((NRows,NImages))


# Importing an HR ground-truth image, and simulating the LR observations
#----------------------------------------------------------------
img = torchvision.io.read_image('Dataset/image.jpg')
img = T.Resize(size=300)(img)
img = T.Grayscale()(img)

set_img, img, img_rescaled, padding = set_img_lr(img, parameters)


# Form the set of linear equations from every pairwise image correspondence in the set

RowIndex = 0

for i in range(NImages):
    for j in range(i+1, NImages):

        img1 = convert(set_img[i], 0, 255, torch.uint8)
        img2 = convert(set_img[j], 0, 255, torch.uint8)

        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(np.array(img1), None)
        kp2, des2 = orb.detectAndCompute(np.array(img2), None)

        # matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        p1 = []
        p2 = []
        for match in matches[:100]:
            p1.append(list(kp1[match.queryIdx].pt))
            p2.append(list(kp2[match.trainIdx].pt))

        p1 = np.array(p1)
        p2 = np.array(p2)

        p1 = torch.tensor(p1)
        p2 = torch.tensor(p2)
        p1 = torch.unsqueeze(p1,0)
        p2 = torch.unsqueeze(p2,0)
        

        # h, status = cv2.findHomography(p1, p2, cv2.RANSAC)

        T1, T2 = np.array(pytorch3d.ops.corresponding_points_alignment(p1,p2)[1][0])
        '''
        **X**: Batch of `d`-dimensional points
            of shape `(minibatch, num_points_X, d)` or a `Pointclouds` object.
        **Y**: Batch of `d`-dimensional points
            of shape `(minibatch, num_points_Y, d)` or a `Pointclouds` object.
        '''

        # aligned_image = cv2.warpPerspective(destination_image, h, (source_image.shape[1], source_image.shape[0]))
        A[RowIndex][ j] = 1
        A[RowIndex][ i] = -1
        TX[RowIndex] = T1
        TY[RowIndex] = T2

        RowIndex += 1

dXHat = np.linalg.lstsq(A, TX, rcond=None)[0]
dYHat = np.linalg.lstsq(A, TY, rcond=None)[0]

dXHat = S*(dXHat - dXHat[0])
dYHat = S*(dYHat - dYHat[0])
dXHat = [int(round(i)) for i in dXHat[:,0]]
dYHat = [int(round(i)) for i in dYHat[:,0]]

print(dXHat, dYHat)


# Create a Set of LR images

_, H, W = img.shape
h, w = set_img[0].shape

Fusion_img = np.zeros((H+2*padding, W+2*padding))

for k in range(NImages):
    LR_img = set_img[k]
    for i in range(h):
        for j in range(w):
            px = i*S + dx[k] + padding
            py = j*S + dy[k] + padding
            Fusion_img[px][py] = LR_img[i][j]

Fusion_img = torch.tensor(Fusion_img)


#----------------------------------------------------------------
# _, ax = plt.subplots(ncols= 2)

# img = T.ToPILImage()(img.to('cpu'))
# ax[0].imshow(np.asarray(img), cmap = 'gray')
# ax[0].axis('off')
# ax[0].set_title('Original_img')

# img = T.ToPILImage()(Fusion_img.to('cpu'))
# ax[1].imshow(np.asarray(img), cmap = 'gray')
# ax[1].axis('off')
# ax[1].set_title('Fusion_img')

# plt.show()

_, ax = plt.subplots(ncols= NImages + 1)

img = T.ToPILImage()(Fusion_img.to('cpu'))
ax[0].imshow(np.asarray(img), cmap = 'gray')
ax[0].axis('off')
ax[0].set_title('Fusion_img')

# img = T.ToPILImage()(img_rescaled.to('cpu'))
# ax[0].imshow(np.asarray(img), cmap = 'gray')
# ax[0].axis('off')
# ax[0].set_title('img_rescaled')

for k in range(NImages):
    img = T.ToPILImage()(set_img[k].to('cpu'))
    ax[k+1].imshow(np.asarray(img), cmap = 'gray')
    ax[k+1].axis('off')
    ax[k+1].set_title('LR ' + str(k+1))

plt.show()