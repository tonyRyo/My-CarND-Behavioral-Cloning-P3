import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split

correction = 0.2
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

#Split the data set into train_samples and validation_samples
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print(len(train_samples))
print(len(validation_samples))

def generator(samples, batch_size=32):
    num_samples = len(samples)

    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name0 = './data/IMG/'+batch_sample[0].split('/')[-1]
                nameL = './data/IMG/'+batch_sample[1].split('/')[-1]
                nameR = './data/IMG/'+batch_sample[2].split('/')[-1]
                #read in center/left/right camera data and steering angle
                image0 = cv2.imread(name0)
                angle0 = float(batch_sample[3])
                if not image0 is None:
                    image0 = cv2.cvtColor(image0,cv2.COLOR_BGR2RGB)
                    images.append(image0)
                    angles.append(angle0)
                imageL = cv2.imread(nameL)
                if not imageL is None:
                    imageL = cv2.cvtColor(imageL,cv2.COLOR_BGR2RGB)
                    images.append(imageL)
                    angleL = angle0 + correction
                    angles.append(angleL)
                imageR = cv2.imread(nameR)
                if not imageR is None:
                    imageR = cv2.cvtColor(imageR,cv2.COLOR_BGR2RGB)
                    images.append(imageR)
                    angleR = angle0 - correction
                    angles.append(angleR)              

            #flip the image to augment the data set
            X_train = np.array(images)
            y_train = np.array(angles)
            X_train = np.concatenate((X_train,np.fliplr(images)))
            y_train = np.concatenate((y_train,-y_train))
            
            yield (X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
#samples_generator = generator(samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten,Dense,Activation,Convolution2D,MaxPooling2D
from keras.layers import Cropping2D,Lambda,Dropout

#build my model
model = Sequential()
model.add(Lambda(lambda x:x/255-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dropout(0.7))
model.add(Dense(100))
model.add(Dropout(0.7))
model.add(Dense(50))
model.add(Dropout(0.7))
model.add(Dense(10))
model.add(Dense(1))
print(model.summary())

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6, validation_data=validation_generator,
            nb_val_samples=int(len(validation_samples)*6), nb_epoch=3, verbose=1)

model.save('model.h5')

#from keras.utils.vis_utils import plot_model
#plot_model(model, to_file='model.png')

### print the keys contained in the history object
print(history_object.history.keys())
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

