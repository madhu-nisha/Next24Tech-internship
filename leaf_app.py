import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
import random
import os
from os import listdir
from PIL import Image
from sklearn.preprocessing import label_binarize,  LabelBinarizer
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dropout, Dense
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from tensorflow.keras.utils import to_categorical
import flask
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("C:/Users/Madhunisha/Downloads/Leaf/Leaf/leaf.h5")

#Converting Images to array
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, (256,256))
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

# Define class labels
all_labels = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

# Define the image processing function
def process_image(image):
    image = image.resize((256, 256))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0) / 225.0
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        # Process the image
        image = Image.open(file)
        processed_image = process_image(image)

        # Make a prediction
        prediction = model.predict(processed_image)
        predicted_class = all_labels[np.argmax(prediction)]

        # Return prediction as JSON
        return jsonify({'prediction': predicted_class})

if __name__ == "__main__":
    app.run(debug=True)







