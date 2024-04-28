from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
import pickle

app = Flask(__name__)

model = tf.keras.models.load_model(r'C:\Users\gauri\OneDrive\Desktop\DeepFake\deepfake_10_model.h5')

def preprocess_image(image):
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image

def predict(image):
    image = preprocess_image(image)
    prediction = model.predict(np.expand_dims(image, axis=0))[0]
    return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    if request.method == 'POST':
        # Check if an image file is uploaded
        if 'file' not in request.files:
            return render_template('index.html', message='')
        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return render_template('index.html', message='')

        # If the file is valid
        if file:
            image = tf.image.decode_image(file.read(), channels=3)
            prediction = predict(image)
            predicted_class = np.argmax(prediction)
            if predicted_class == 0:
                result = 'Deepfake'
            else:
                result = 'Real'
            return render_template('index.html', message='Prediction: {} || Result: {}'.format(prediction,result))

if __name__ == '__main__':
    app.run(debug=True)
