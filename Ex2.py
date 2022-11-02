#packages
exec(open('packages.py').read())

#parameters
from Ex1 import parameters

# Importing an HR ground-truth image, and simulating the LR observations
#----------------------------------------------------------------
img = torchvision.io.read_image('Dataset/image.jpg')
img = T.Resize(size=300)(img)
img = T.Grayscale()(img)


set_img, img, img_rescaled = set_img_lr(img, parameters)

# ----------------------------------------------------------------
# img = T.ToPILImage()(set_img[0].to('cpu'))
# plt.imshow(np.asarray(img), cmap = 'gray')
# plt.show()

_, ax = plt.subplots(ncols= parameters['NImages'] + 1 )

img = T.ToPILImage()(img.to('cpu'))
ax[0].imshow(np.asarray(img), cmap = 'gray')
ax[0].axis('off')
ax[0].set_title('Original')

for k in range(parameters['NImages']):
    img = T.ToPILImage()(set_img[k].to('cpu'))
    ax[k+1].imshow(np.asarray(img), cmap = 'gray')
    ax[k+1].axis('off')
    ax[k+1].set_title('LR ' + str(k+1))

plt.show()