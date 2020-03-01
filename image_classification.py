
#Import relevant libraries
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import pandas as pd
import numpy as np
from PIL import Image
from pylab import *
import glob
import tensorflow

#Import the images. Images are in the folder images_laptop, in the working directory
image_list = []
for filename in glob.glob('images_laptop/*'):
    im=Image.open(filename)
    image_list.append(im)

#Convert all the images into grayscale
grey_image = []
for image in image_list:
    grey_image.append(image.convert('L'))

#Resize images to 256X256
resized_images = []
for image in grey_image:
    resized_images.append(image.resize((256,256)))

#Convert images to array
image_array = []
for image in resized_images:
    image_array.append(array(image))

# np.asarray(image_array[0]).shape
#Reshape for keras
image_array = np.asarray(image_array).reshape(1000,256,256,1)

# Split the data between train and test
x_train = np.asarray(list(image_array[0:450]) + list(image_array[500:950]))

x_test = np.asarray(list(image_array[450:500]) + list(image_array[950:1000]))
y_train = np.append(np.repeat(1,450),np.repeat(2,450))
y_test = np.append(np.repeat(1,50),np.repeat(2,50))
# y_train.shape
# y_train.shape

#Split train-test
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)


#CNN Model
CNNmodel = Sequential()
CNNmodel.add(Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=(256, 256, 1)))
CNNmodel.add(Conv2D(64, (3, 3), activation='relu'))
CNNmodel.add(MaxPooling2D(pool_size=(2, 2)))
CNNmodel.add(Conv2D(64, (3, 3), activation='relu'))
CNNmodel.add(MaxPooling2D(pool_size=(2, 2)))
CNNmodel.add(Flatten())
CNNmodel.add(Dense(128, activation='relu'))
CNNmodel.add(Dense(3, activation='softmax'))
model = keras.models.Model(inputs=CNNmodel.input,outputs=CNNmodel._output_layers[0].output)

#Store the features for ML Models
feat_train = model.predict(x_train)
feat_test = model.predict(x_test)
#Train CNN
CNNmodel.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
CNNmodel.fit(x_train, y_train, validation_data=(x_test, y_test))
#Evaluate
performance = CNNmodel.evaluate(x_test, y_test)
print('Test accuracy:', performance[1])

#Train and evaluate SVM
from sklearn.svm import SVC
svm = SVC(kernel='rbf')
svm.fit(feat_train,np.argmax(y_train,axis=1))

print("Test Accuracy", svm.score(feat_test,np.argmax(y_test,axis=1)))

#Train and evaluate RF
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(feat_train,np.argmax(y_train,axis=1))

rf.score(feat_test,np.argmax(y_test,axis=1))


#Also try Light GBM
import lightgbm

lgbm = lightgbm.LGBMClassifier()
lgbm.fit(feat_train,np.argmax(y_train,axis=1))

lgbm.score(feat_test,np.argmax(y_test,axis=1))



#Train and evaluate ANN (or DNN)
model_ann = Sequential()
model_ann.add(Dense(32, input_shape=(256, 256, 1), activation='relu'))

model_ann.add(Dense(32, activation='relu'))

model_ann.add(Flatten())
model_ann.add(Dense(3, activation='softmax'))

model_ann.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

history = model_ann.fit(x_train, y_train,epochs=1,batch_size=30)
print(model_ann.evaluate(x_test,y_test)[1])

