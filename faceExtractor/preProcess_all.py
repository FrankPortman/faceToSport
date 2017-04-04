import cv2
from matplotlib import pyplot as plt
import numpy as np
import matplotlib # matplotlib.pyplot.show('hold')
import os

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

## basic keras model 
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

train = np.vstack(scaled_imagesP)
encoder = LabelEncoder()
encoder.fit(labels)
encoded_lab = encoder.transform(labels)

labs = keras.utils.np_utils.to_categorical(encoded_lab, 3)

train = train.astype("float32")
train /= 255

model = Sequential()
model.add(Dense(1200, activation='relu', input_shape=(40000, )))
model.add(Dropout(0.02))
model.add(Dense(2400, activation='tanh'))
model.add(Dropout(0.02))
model.add(Dense(2400, activation='sigmoid'))
model.add(Dropout(0.02))
model.add(Dense(2400, activation='relu'))
model.add(Dropout(0.02))
model.add(Dense(512, activation='tanh'))
model.add(Dropout(0.02))
model.add(Dense(2400, activation='tanh'))
model.add(Dropout(0.02))
model.add(Dense(2400, activation='sigmoid'))
model.add(Dropout(0.02))
model.add(Dense(2400, activation='relu'))
model.add(Dropout(0.02))
model.add(Dense(512, activation='tanh'))
model.add(Dropout(0.02))
model.add(Dense(3, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


history = model.fit(train, labs, batch_size=10, nb_epoch=10, verbose=1, class_weight='auto')

out = model.predict(train)

out = [i.tolist().index(max(i)) for i in out]

np.mean(out)

from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train.shape

scaled_images = []

for i in images:
    top, bottom, left, right = max_x - i.shape[0], 0, int(np.floor((max_y - i.shape[1]) / 2.0)), int(np.ceil((max_y - i.shape[1]) / 2.0))
    scaled_images.append(cv2.copyMakeBorder(i, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255, 255)))
    
    
scaled_images = [cv2.resize(i, (28, 28)) for i in scaled_images]
scaled_images = [cv2.resize(i, (28, 28)) for i in images]

images_array = np.dstack(scaled_images)
images_array = np.rollaxis(images_array, -1)

images_array.shape

plt.imshow(images_array[0], cmap='gray')


# Simple CNN model for CIFAR-10
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

images_array = np.expand_dims(images_array, 4)

model = Sequential()
model.add(Convolution2D(12, 2, 2, input_shape=(28, 28, 1), activation='relu'))
model.add(Dropout(0.1))
model.add(Convolution2D(12, 2, 2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(20, 3, 3, activation='relu'))
model.add(Convolution2D(20, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Convolution2D(12, 2, 2, activation='relu'))
model.add(Convolution2D(20, 2, 2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.1))
model.add(Dense(3, activation='softmax'))
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
print(model.summary())

history = model.fit(images_array, labs, batch_size=20, nb_epoch=1000, verbose=1, class_weight='auto')

out = model.predict(images_array)

out

out2 = [i.tolist().index(max(i)) for i in out]

np.mean(out2)

from sklearn.metrics import log_loss

log_loss(labs, out)

losses = [-(sum(np.log(x) * y)) for x,y in zip(out, labs)]
losses.index(max(losses))

plt.imshow(scaled_images[2511], cmap='gray')  

out[2511]
labs[2511]
labels[2511]

worst_to_best = np.flipud(np.argsort(losses))
worst_to_best.tolist().index(2511)

plt.imshow(scaled_images[worst_to_best[0]], cmap='gray')
out[worst_to_best[0]]
labs[worst_to_best[0]]
labels[worst_to_best[0]]

plt.imshow(scaled_images[worst_to_best[-1]], cmap='gray')
out[worst_to_best[-1]]
labs[worst_to_best[-1]]
labels[worst_to_best[-1]]

plt.imshow(scaled_images[worst_to_best[14]], cmap='gray')
out[worst_to_best[14]]
labs[worst_to_best[14]]
labels[worst_to_best[14]]

def plotFace(idx):
    plt.imshow(scaled_images[worst_to_best[idx]], cmap='gray')
    print 'predictions are', out[worst_to_best[idx]]
    print 'labels are', labs[worst_to_best[idx]]
    print 'sport is', labels[worst_to_best[idx]]
    

plotFace(13)
plotFace(14)
plotFace(15)    
plotFace(16)
plotFace(17)
plotFace(18)
plotFace(19)
plotFace(1600)
plotFace(2000)

accuracies = history.history.get('acc')
epochs = history.epoch

plt.plot(epochs, accuracies)
