#packages
exec(open('packages.py').read())

#function
from tkinter import N
from Ex2 import set_img_lr

#random coordinate to simulate a motion 
def random_coor(n):
    x = []
    y = []
    h = int(np.sqrt(n))
    i = 0
    count = 0
    while(i<n):
        temp_x = int(np.random.rand()*100 % (h+ int(count/100)))
        temp_y = int(np.random.rand()*100 % (h+ int(count/100)))
        x.append(temp_x)
        y.append(temp_y)
        for j in range(len(x) - 1):
            if temp_x == x[j] and temp_y == y[j]:
                x.pop()
                y.pop()
                i -= 1
                count += 1
                break
        i += 1
    return x, y


#Parameters
S = 3
NImages = 9
dx, dy = random_coor(NImages)
print(dx, dy)
# NImages = 4
# dx = [0, 0, 1, 1]
# dy = [0, 1, 0, 1]

# NImages = 5
# dx = [0, 2, 1, 1, 2] 
# dy = [0, 0, 1, 2, 1]

# NImages = 7
# dx = [0, 2, 1, 1, 2, 0, 2] 
# dy = [0, 0, 1, 2, 1, 2, 2]

# NImages = 9
# dx = [0, 2, 1, 1, 2, 0, 2, 0, 0] 
# dy = [0, 0, 1, 2, 1, 2, 2, 1, 2]

NoiseStd = 0
K = [[1]]
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
Flag_img = np.ones((H, W))

for k in range(NImages):
    LR_img = set_img[k]
    for i in range(h):
        for j in range(w):
            px = i*S + dx[k]
            py = j*S + dy[k]
            Fusion_img[px][py] = LR_img[i][j]
            Flag_img[px][py] = 0


SR_img = inpaint.inpaint_biharmonic(Fusion_img, Flag_img)

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