#packages
exec(open('packages.py').read())


#Parameters
S = 3
NImages = 9
dx, dy = random_coor(NImages)
print(dx, dy)

NoiseStd = 10/255
K = gkern(2.5)
#[[1, 2, 1], [2, 4, 2], [1, 2, 1]]/16

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
set_img, img = set_img_lr(img, parameters)

H, W = img.shape
h, w = set_img[0].shape

Fusion_img = np.zeros((H, W))

for k in range(NImages):
    LR_img = set_img[k]
    for i in range(h):
        for j in range(w):
            px = i*S + dx[k]
            py = j*S + dy[k]
            Fusion_img[px][py] = LR_img[i][j]


# SR_img = richardson_lucy(Fusion_img, K, num_iter=50)
SR_img = deconv2DTV(Fusion_img, K)

_, ax = plt.subplots(nrows = 2, ncols=3)

ax[0][0].imshow(img, cmap = 'gray')
ax[0][0].axis('off')
ax[0][0].set_title('Original')

ax[0][1].imshow(SR_img, cmap = 'gray')
ax[0][1].axis('off')
ax[0][1].set_title("SR")

ax[0][2].imshow(Fusion_img, cmap = 'gray')
ax[0][2].axis('off')
ax[0][2].set_title("Fusion")

for k in range(3):
    ax[1][k].imshow(set_img[k], cmap = 'gray')
    ax[1][k].axis('off')
    ax[1][k].set_title("LR " + str(k+1))

plt.show()