#packages
exec(open('packages.py').read())

#Setup the variables

S = 3
NImages = 6
dx, dy = random_coor(NImages)
print(dx, dy)
# dx = [0, 0, 1, 1]
# dy = [0, 1, 0, 1]


NoiseStd = 0
K = [[1]]
# K = gkern(2.5)
# K = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]/16
K = np.array(K)

NRows = math.comb(NImages, 2)
TX = np.zeros((NRows,1))
TY = np.zeros((NRows,1))
A = np.zeros((NRows,NImages))


parameters = {}
parameters['S'] = S
parameters['NImages'] = NImages
parameters['dx'] = dx
parameters['dy'] = dy
parameters['NoiseStd'] = NoiseStd
parameters['K'] = K


def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

# Importing an HR ground-truth image, and simulating the LR observations
#----------------------------------------------------------------
img = io.imread('Dataset/image.jpg', as_gray=True)
img = rescale(img, 0.5, anti_aliasing=False)

# Create a Set of LR images
set_img, img = set_img_lr(img, parameters)


# Form the set of linear equations from every pairwise image correspondence in the set

RowIndex = 0

for i in range(NImages):
    for j in range(i+1, NImages):

        img1 = convert(set_img[0], 0, 255, np.uint8)
        img2 = convert(set_img[1], 0, 255, np.uint8)

        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        p1 = []
        p2 = []
        for match in matches:
            p1.append(list(kp1[match.queryIdx].pt))
            p2.append(list(kp2[match.trainIdx].pt))

        p1 = np.array(p1)
        p2 = np.array(p2)

        h, status = cv2.findHomography(p1, p2, cv2.RANSAC)

        # aligned_image = cv2.warpPerspective(destination_image, h, (source_image.shape[1], source_image.shape[0]))
        A[RowIndex][ j] = 1
        A[RowIndex][ i] = -1
        TX[RowIndex] = h[0,2]
        TY[RowIndex] = h[1,2]

        RowIndex += 1

dXHat = np.linalg.lstsq(A, TX, rcond=None)[0]
dYHat = np.linalg.lstsq(A, TY, rcond=None)[0]

dXHat = S*(dXHat - dXHat[0])
dYHat = S*(dYHat - dYHat[0])
print(dXHat, dYHat)
# draw first 50 matches
# match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:5], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# cv2.imshow('Matches', match_img)
# cv2.waitKey()


# _, ax = plt.subplots(ncols = 6)

# ax[0].imshow(img)
# ax[0].axis('off')
# ax[0].set_title("img")


# # for k in range(5):
# #     m = k+1
# #     ax[m].imshow(set_img[k])
# #     ax[m].axis('off')
# #     ax[m].set_title("LR " + str(k+1))

# plt.show()