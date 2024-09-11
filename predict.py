import os
import argparse
import numpy as np
import json
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
parser = argparse.ArgumentParser(description='Predict Flower Class')
parser.add_argument("-f", "--filepath",  type=str, help='Filepath (e.g. ./test_images/cautleya_spicata.jpg)')
parser.add_argument("-m", "--model" , type=str, help='Model (e.g mymodel_1587052746.h5')
parser.add_argument("-nc", "--top_K", type=int, help='Number of classes to show')
arg = parser.parse_args()
print (arg.filepath, arg.model, arg.top_K)
#get the arguments
image = arg.filepath
model = arg.model
top_K= arg.top_K
# load the model
reloaded_keras_model = tf.keras.models.load_model('1589095413.h5', custom_objects={'KerasLayer':hub.KerasLayer})
reloaded_keras_model.summary()
#get class names
with open('label_map.json', 'r') as f:
    class_names = json.load(f)
# the process_image function

def process_image(prediction_image):
    prediction_image = tf.cast(prediction_image, tf.float32)
    prediction_image = tf.image.resize(prediction_image, (224, 224))
    prediction_image /= 255
    prediction_image= prediction_image.numpy().squeeze()
    return prediction_image

# the predict function
def predict(image_path, model, top_k):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    prediction = reloaded_keras_model.predict(np.expand_dims(processed_test_image, axis=0))
    top_values, top_indices = tf.math.top_k(prediction, top_k)
    print("These are the top propabilities",top_values.numpy()[0])
    top_classes = [class_names[str(value+1)] for value in top_indices.cpu().numpy()[0]]
    print('Of these top classes', top_classes)
    return top_values.numpy()[0], top_classes
# # the prediction
top_values, top_classes = predict(image, model, top_K)