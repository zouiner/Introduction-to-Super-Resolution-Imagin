exec(open('packages.py').read())

# Set parameters for the degraded HR
#----------------------------------------------------------------

# Define the SR magnification
S = 2
NImages = 4
dx, dy = random_coor(NImages)
print(dx, dy)

NoiseStd = 0/255
K = gkern(2.5)
# K = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]/16
K = [[1]]
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

