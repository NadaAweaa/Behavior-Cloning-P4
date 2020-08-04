## Importing Libraries 
import os
import csv
import cv2
import numpy as np
import random
import math

import tensorflow as tf
from keras.layers.convolutional import Convolution2D
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout, Cropping2D
from keras.models import Sequential
from keras.optimizers import Adam
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from zipfile import ZipFile


def uncompress_folder(dir,name):
    if(os.path.isdir(name)):
        print('Data extracted')
    else:
        with ZipFile(dir) as zipf:
            zipf.extractall('data4')
#uncompress_folder('data.zip','data4')

samples = []

with open('./data4/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

samples = samples[1:]
   
train_samples, validation_samples = train_test_split(samples,test_size=0.2)
    
    
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle( samples )
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data4/data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                image_flipped = np.copy( np.fliplr(center_image))
                #center_image = np.expand_dims(center_image, axis=0)
                center_angle = float(batch_sample[3])
                angle_flipped = -center_angle
                images.append(center_image)
                images.append(image_flipped)
                angles.append(center_angle)
                angles.append(angle_flipped)
                

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            print(X_train.shape)
            yield sklearn.utils.shuffle(X_train, y_train)    

    
# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
    
#input_shape=(160, 320, 3)
ch, row, col = 3, 16, 320  # Trimmed image format

model = Sequential()
#model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))
#model.add(Lambda(lambda x: x / 127.5 - 1.0,  input_shape=(ch, row, col),output_shape=(ch, row, col)))
#model.add(Cropping2D(cropping=((70,24),(60,60))))
model.add( Cropping2D( cropping=((70,25), (60,60)), input_shape=(160,320,3)))
model.add( Lambda(lambda x: x/255. - 0.5 ))

#x = np.arange(np.prod(input_shape)).reshape(input_shape)
#model.add(Cropping2D(cropping=((70,25),(0,0))))(x) 
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="elu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="elu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="elu"))
model.add(Conv2D(64, (3, 3), activation="elu"))
model.add(Conv2D(64, (3, 3), activation="elu"))
model.add(Dropout(0.8))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')


history_object = model.fit_generator(train_generator, steps_per_epoch= len(train_samples)/batch_size ,
                                     validation_data=validation_generator,
                                     validation_steps= len(validation_samples)/batch_size,
                                     epochs=5, 
                                     verbose=1)

model.save( 'model.h5' )
print('model saved!')

print(history_object.history.keys())

### plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()