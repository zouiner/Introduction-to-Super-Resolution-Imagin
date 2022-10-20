#packages
exec(open('packages.py').read())


#Parameters
S = 3
NImages = 9
dx, dy = random_coor(NImages)
print(dx, dy)

NoiseStd = 2/255
size_K = 5
K = gaussuian_filter(size_K)
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

H, W = img.shape

# Create a Set of LR images
set_img = set_img_lr(img, parameters)

h, w = set_img[0].shape

Fusion_img = np.zeros((H, W))

for k in range(NImages):
    LR_img = set_img[k]
    for i in range(h):
        for j in range(w):
            px = i*S + dx[k]
            py = j*S + dy[k]
            Fusion_img[px][py] = LR_img[i][j]


SR_img, _ = unsupervised_wiener(Fusion_img, K)

_, ax = plt.subplots(ncols=3)

ax[0].imshow(img, cmap = 'gray')
ax[0].axis('off')
ax[0].set_title('Original')

ax[1].imshow(SR_img, cmap = 'gray')
ax[1].axis('off')
ax[1].set_title("SR")

for k in range(1):
    ax[k+2].imshow(set_img[k], cmap = 'gray')
    ax[k+2].axis('off')
    ax[k+2].set_title("Blurred" + str(k+1))

plt.show()