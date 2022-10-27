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
set_img, img, img_rescaled = set_img_LR_CFA(img, parameters)


# _, ax = plt.subplots(ncols=2)

# ax[0].imshow(img)
# ax[0].axis('off')
# ax[0].set_title('Original')

# ax[1].imshow(set_img[0])
# ax[1].axis('off')
# ax[1].set_title("LR")

# plt.show()