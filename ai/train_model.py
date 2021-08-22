import matplotlib
from numpy.core.fromnumeric import shape
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from tensorflow.keras.utils import to_categorical
# from utils.lenet import LeNet
from tensorflow.keras.models import Sequential
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

import tensorflow as tf

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
    help="path to output model")
ap.add_argument("-e", "--epochs", type=int, default=140, required=False,
    help="path to output loss/accuracy plot")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
    help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initia learning rate,
# and batch size
EPOCHS = args["epochs"]
INIT_LR = 1e-3
BS = 32

# initialize the data and labels
# print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    label = 1 if label == "daytime" else 0
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
    labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
# trainY = to_categorical(trainY, num_classes=2)
# testY = to_categorical(testY, num_classes=2)

print(shape(trainX))
trainX.reshape(-1, 224, 224, 1)
print(shape(trainY))

# print(trainX)

# construct the image generator for data augmentation
# aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
#     height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
#     horizontal_flip=True, fill_mode="nearest")

# aug.fit(trainX)

# initialize the model
# print("[INFO] compiling model...")
model = Sequential() # LeNet.build(width=28, height=28, depth=3, classes=2)

model.add(Dense(units=16, input_shape=(224,224,3), activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=2, activation='softmax'))

model.summary()

opt = Adam(learning_rate=0.000001)

# opt = adam_v2(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])
# # model.compile(loss="binary_crossentropy", optimizer="Adam", self=None)

# # train the network
# print("[INFO] training network...")
# H = model.fit(aug.flow(trainX, trainY, batch_size=BS),
#     validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
#     epochs=EPOCHS, verbose=1)

H = model.fit(trainX, trainY, epochs = 100)

# # save the model to disk
# # print("[INFO] serializing network...")
# model.save(args["model"])

# # plot the training loss and accuracy
# plt.style.use("ggplot")
# plt.figure()
# N = EPOCHS
# plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")

# plt.title("Training Loss and Accuracy on Day/Night")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")
# plt.savefig(args["plot"])