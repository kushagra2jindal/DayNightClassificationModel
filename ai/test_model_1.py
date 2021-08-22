from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

test_path = '/Users/kushagra/Desktop/DayNightClassificationModel/data_collection/test_data'
model_path = '/Users/kushagra/Desktop/DayNightClassificationModel/models/day_night_1.h5'

test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path, target_size=(224,224), classes=['daytime', 'night'], batch_size=10, shuffle=False)
test_img, test_labels = next(test_batches)

# load model and predict
model = load_model(model_path)
predictions = model.predict(x=test_batches, batch_size=10, verbose=0)

rounded_predictions = np.argmax(predictions, axis=-1)
rounded_labels = np.argmax(test_labels, axis=-1)
cm = confusion_matrix(y_true=rounded_labels, y_pred=rounded_predictions)
print(cm)