import cv2
from matplotlib import pyplot as plt 

images = []

for file in os.listdir('../imgs/Extracted/'):
    img = cv2.imread(os.path.join('../imgs/Extracted', file), 0)
    if img is not None:
        images.append((img, file[0:3]))


labels = [i[1] for i in images]
images = [i[0] for i in images]
## append whitespace after finding the max size then resize
max_x, max_y = max([i.shape for i in images])

max_x = max([i.shape[0] for i in images])
max_y = max([i.shape[1] for i in images])

scaled_images = []

for i in images:
    top, bottom, left, right = max_x - i.shape[0], 0, int(np.floor((max_y - i.shape[1]) / 2.0)), int(np.ceil((max_y - i.shape[1]) / 2.0))
    scaled_images.append(cv2.copyMakeBorder(i, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255, 255)))
    
    
scaled_images = [cv2.resize(i, (200, 200)) for i in scaled_images]
scaled_imagesP = [i.flatten() for i in scaled_images]

all([len(i) == (200 * 200) for i in scaled_imagesP])

plt.imshow(scaled_images[0], cmap='gray')
plt.imshow(scaled_images[84], cmap='gray')
plt.imshow(scaled_images[1024], cmap='gray')

[i.shape[0] for i in images].index(max_x)

plt.imshow(scaled_images[405], cmap='gray')
plt.imshow(scaled_images[2504], cmap='gray')

kek = sum(images)
kek = kek / len(images)

plt.imshow(kek, cmap='gray')

kek = np.ndarray(images)
kek = kek.astype("double")

kek + kek

images[0].dtype

kek = [i.astype("double") for i in scaled_images]
kek = sum(kek)
kek = kek / len(images)

plt.imshow(kek, cmap='gray')


### average faces
scaled_images = [i.astype("double") for i in scaled_images]

def plotAvgFace(league):
    mat = sum([scaled_images[i] for i,x in enumerate(labels) if x == league])
    mat = mat / len([i for i in labels if i == league])
    plt.imshow(mat, cmap='gray')
    

nhl = sum([scaled_images[i] for i,x in enumerate(labels) if x == 'nhl'])
nhl = nhl / len([i for i in labels if i == 'nhl'])

plt.imshow(nhl, cmap='gray')

plotAvgFace("nhl")
plotAvgFace("nfl")
plotAvgFace("nba")
