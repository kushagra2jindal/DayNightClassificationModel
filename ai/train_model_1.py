import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


train_path = "/Users/kushagra/Desktop/DayNightClassificationModel/data_collection/data"
test_path = "/Users/kushagra/Desktop/DayNightClassificationModel/data_collection/test_data"

# 224, 224 is the height and width of the images
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path, target_size=(224,224), classes=['daytime', 'night'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path, target_size=(224,224), classes=['daytime', 'night'], batch_size=10, shuffle=False)

test_img, test_labels = next(test_batches)
print(test_labels)

model = Sequential([
    # 224*224 is the height and width and 3 is color RGB
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(224,224,3)),
    # Divide our images into 2
    MaxPool2D(pool_size=(2, 2), strides=2),
    # Convolutional Layer
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    # Flat everything into one Tensorflow dimention
    Flatten(),
    # Softmax will give us probability 
    Dense(units=2, activation='softmax')
])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.001), loss= 'categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_batches, epochs=10, verbose=2)
# print(labels)

predictions = model.predict(x=test_batches, batch_size=10, verbose=0)

rounded_predictions = np.argmax(predictions, axis=-1)
rounded_labels = np.argmax(test_labels, axis=-1)

for i in rounded_labels:
    print(i)

# print (predictions)

cm = confusion_matrix(y_true=rounded_labels, y_pred=rounded_predictions)
cm_plot_labels = ['daytime','night']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')


# Save model

model.save('/Users/kushagra/Desktop/DayNightClassificationModel/models/day_night_1.h5')

# print(train_batches.n)
