import cv2
from matplotlib import pyplot as plt 

images = []

for file in os.listdir('../imgs/Extracted/'):
    img = cv2.imread(os.path.join('../imgs/Extracted', file), 0)
    if img is not None:
        images.append(img)
    

images = [cv2.resize(i, (200, 200)) for i in images]
imagesP = [i.flatten() for i in images]

all([len(i) == (200 * 200) for i in imagesP])

plt.imshow(images[0], cmap='gray')
plt.imshow(images[84], cmap='gray')
plt.imshow(images[1024], cmap='gray')

kek = sum(images)
kek = kek / len(images)

plt.imshow(kek, cmap='gray')

kek = np.ndarray(images)
kek = kek.astype("double")

kek + kek

images[0].dtype

kek = [i.astype("double") for i in images]
kek = sum(kek)
kek = kek / len(images)

plt.imshow(kek, cmap='gray')