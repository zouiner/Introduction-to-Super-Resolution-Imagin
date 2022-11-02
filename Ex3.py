#packages
exec(open('packages.py').read())


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
H, W = img.shape

h, w = set_img[0].shape

SR_img = np.zeros((H, W))


for k in range(NImages):
    LR_img = set_img[k]
    for i in range(h):
        for j in range(w):
            px = i*S + dx[k]
            py = j*S + dy[k]
            SR_img[px][py] = LR_img[i][j]



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