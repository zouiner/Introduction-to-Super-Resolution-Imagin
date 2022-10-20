#packages
exec(open('packages.py').read())

#parameters
from Ex1 import parameters

# Importing an HR ground-truth image, and simulating the LR observations
#----------------------------------------------------------------
img = io.imread('Dataset/image.jpg', as_gray=True)
img = rescale(img, 0.25, anti_aliasing=False)


set_img = set_img_lr(img, parameters)

# io.imshow_collection(set_img, cmap= 'gray')
# plt.show()
# ----------------------------------------------------------------
# _, ax = plt.subplots(ncols=2)

# ax[0].imshow(img, cmap = 'gray')
# ax[0].axis('off')
# ax[0].set_title('Original')

# ax[1].imshow(set_img[0], cmap = 'gray')
# ax[1].axis('off')
# ax[1].set_title("Blurred")

# plt.show()