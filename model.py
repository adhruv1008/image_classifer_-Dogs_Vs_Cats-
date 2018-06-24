#importing libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import cv2
import numpy as np

K.set_image_dim_ordering('th')

#creating training data and validation data out of training and validation images in jpg format
batch_size=16
train_datagen = ImageDataGenerator(1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
val_datagen = ImageDataGenerator(1./255)

train_generator = train_datagen.flow_from_directory('data/train',target_size=(150,150),batch_size=batch_size,class_mode='binary')
val_generator = val_datagen.flow_from_directory('data/validation',target_size=(150,150),batch_size=batch_size,class_mode='binary')



#Building a neural network
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch=200 // batch_size,
        epochs=10,
        validation_data=val_generator,
        validation_steps=80 // batch_size)
model.save_weights('first_try.h5')
model.load_weights('first_try.h5')

#Loading a test image , to check the working of the model, to test on new image just place the image in the same directory as the dog.jpg and replace
#dog.jpg in the imread to the new image file name

#model predicts 1 for dog and 0 for cat

img = cv2.imread('dog.jpg')
img = cv2.resize(img,(150,150))
img = np.array(img)
img = img.reshape(1,3,150,150)

#displaying the test image
disp_img = cv2.imread('dog.jpg',1)
cv2.imshow('image',disp_img)

#printing the prediction
print (model.predict(img))