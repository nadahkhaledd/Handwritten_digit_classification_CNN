from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import numpy as np
import cv2

(trainX, trainY), (testX, testY) = mnist.load_data()

for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
# show the figure
plt.show()


trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

trainX= trainX.astype('float32')
testX = testX.astype('float32')
trainX = trainX / 255.0
testX = testX / 255.0

trainX, trainY = shuffle(trainX, trainY, random_state=0)

#print(trainX[0].shape)

model = keras.models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=trainX[0].shape),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

#print(model.summary())

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
fitting = model.fit(trainX, trainY, epochs=10, validation_split= 0.25)
# if validation acc < acc => overfitting (incase of overfitting => dropout layers)
# if validation acc almost = acc => doing well
print(fitting)

(testLoss, testAcc) = model.evaluate(testX, testY)
print("Test loss accuracy: ", testLoss, "\nTest validation accuracy: ", testAcc)


prediction = model.predict(testX)
prediction_arr = [np.argmax(i) for i in prediction]
print(prediction_arr[:6])

for i in range(6):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(testX[i], cmap=plt.get_cmap('gray'))
# show the figure
plt.show()

#add image and predict
