#packages
exec(open('packages.py').read())

#parameters
exec(open('Ex1.py').read())

# Importing an HR ground-truth image, and simulating the LR observations
#----------------------------------------------------------------
img = io.imread('Dataset/image.jpg', as_gray=True)
img = rescale(img, 0.25, anti_aliasing=False)

def bilinear(img, px, py):
    matA = np.array([[0.5, 0.5]])
    x1 = int(np.ceil(px))
    y1 = int(np.ceil(py))
    x2 = int(np.floor(px))
    y2 = int(np.floor(py))
    try:
        matB = np.array([[img[x1][y1], img[x1][y2]], [img[x2][y1], img[x2][y2]]])
    except:
        matB = np.array([[0, 0], [0, 0]])
    matC = np.array([[0.5], [0.5]])

    ans = np.dot(matA, matB)
    ans = np.dot(ans, matC)

    return ans

def set_img_lr(img, parameters):
    
    S = parameters['S']
    NImages = parameters['NImages']
    dx = parameters['dx']
    dy = parameters['dy']
    NoiseStd = parameters['NoiseStd']
    K = parameters['K']
    
    H, W = img.shape
    h, w = round(H/S), round(W/S)
    img_rescaled = correlate_sparse(img, K)
    
    #Set initial image set
    set_img = []
    for k in range(NImages):
        smallImg = np.zeros((h,w))
        for i in range(h-1):
            for j in range(w-1):
                px = i*S + dx[k] + 0.5
                py = j*S + dy[k] + 0.5
                smallImg[i][j] = bilinear(img_rescaled,px,py) + NoiseStd * np.random.rand()
        set_img.append(smallImg)
    return set_img
    # for k in range(NImages):
    #     smallImg =  resize_local_mean(img_rescaled, (h,w))
    #     for i in range(h):
    #         for j in range(w):
    #             smallImg[i][j] = smallImg[i][j] + NoiseStd * np.random.rand()
    #     set_img.append(smallImg)
    # return set_img

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