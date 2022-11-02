#packages
exec(open('packages.py').read())


#Parameters
S = 2
NImages = 4
dx, dy = random_coor(NImages)
print(dx, dy)
# dx = [0, 0, 1, 1]
# dy = [0, 1, 0, 1]


NoiseStd = 0
K = [[1]]
K = np.array(K)
# K = gkern(2.5)
#[[1, 2, 1], [2, 4, 2], [1, 2, 1]]/16


# CFA = [[1, 0], [2, 1]]
# CFA = [[2, 1], [1, 0]]
# CFA = [[0, 2], [2, 1]]
CFA = [[0, 2], [1, 0]]

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


# Create a SR image
H, W, _ = img.shape
h, w, _ = set_img[0].shape

Fusion_img = np.zeros((H, W, 3))
Flag_img = np.ones((H, W, 3))

for k in range(NImages):
    LR_img = set_img[k]
    for i in range(h):
        for j in range(w):
            px = i*S + dx[k]
            py = j*S + dy[k]
            CFA_c = CFA[i%2][j%2]
            Fusion_img[px][py][CFA_c] = LR_img[i][j][CFA_c]
            Flag_img[px][py][CFA_c] = 0

Fusion_img[:,:,0] = inpaint.inpaint_biharmonic(Fusion_img[:,:,0], Flag_img[:,:,0])
Fusion_img[:,:,1] = inpaint.inpaint_biharmonic(Fusion_img[:,:,1], Flag_img[:,:,1])
Fusion_img[:,:,2] = inpaint.inpaint_biharmonic(Fusion_img[:,:,2], Flag_img[:,:,2])

# SR_img = richardson_lucy(Fusion_img, K, num_iter=50)
SR_img = Fusion_img

_, ax = plt.subplots(nrows = 2, ncols=2)

ax[0][0].imshow(img)
ax[0][0].axis('off')
ax[0][0].set_title('Original')

ax[0][1].imshow(SR_img)
ax[0][1].axis('off')
ax[0][1].set_title("SR")

for k in range(2):
    ax[1][k].imshow(set_img[k], cmap = 'gray')
    ax[1][k].axis('off')
    ax[1][k].set_title("LR " + str(k+1))

plt.show()