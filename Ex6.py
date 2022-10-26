#packages
exec(open('packages.py').read())


#Parameters
S = 2
NImages = 4
# dx, dy = random_coor(NImages)
# print(dx, dy)
dx = [0, 0, 1, 1]
dy = [0, 1, 0, 1]


NoiseStd = 0
K = [[1]]
K = np.array(K)
# K = gkern(2.5)
#[[1, 2, 1], [2, 4, 2], [1, 2, 1]]/16

CFA = [[2, 1], [3, 2]]

parameters = {}
parameters['S'] = S
parameters['NImages'] = NImages
parameters['dx'] = dx
parameters['dy'] = dy
parameters['NoiseStd'] = NoiseStd
parameters['K'] = K
parameters['CFA'] = CFA

# Importing an HR ground-truth image, and simulating the LR observations
#----------------------------------------------------------------
img = cv2.imread('Dataset/image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = rescale(img, 0.25, multichannel=True)

# Create a Set of LR images
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

    for k in range(NImages):
        small_img = np.zeros((h,w,3))
        for i in range(h):
            for j in range(w):
                px = i*S + dx[k] + 0.5
                py = j*S + dy[k] + 0.5
                smallImg[i][j] = bilinear(img_rescaled,px,py) + NoiseStd * np.random.rand()

    return img_rescaled

img = set_img_LR_CFA(img, parameters)




# io.imshow(img)
# plt.show()