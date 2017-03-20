import cv2
from matplotlib import pyplot as plt 

images = []

for file in os.listdir('../imgs/Extracted/'):
    images.append(cv2.imread(os.path.join('../imgs/Extracted/', file), 0))
    

images = [cv2.resize(i, (200, 200)) for i in images]
imagesP = [i.flatten() for i in images]

all([len(i) == (200 * 200) for i in imagesP])

plt.imshow(images[0], cmap='gray')
plt.imshow(images[84], cmap='gray')