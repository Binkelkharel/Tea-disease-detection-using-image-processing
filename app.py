# app.py
from flask import Flask, render_template, request
from matplotlib import pyplot as plt
import os
import numpy as np
import base64
from PIL import Image


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, request, render_template, send_file

app = Flask(__name__)


# @app.route('/upload', methods=['GET','POST'])
# def upload():
#     if request.method == 'GET':        
#         return render_template("upload.html")
#     else:
#         if 'image' in request.files:
#             test_img = request.files['image']

#             context = hello(test_img)

           

@app.route('/output', methods=['POST'])
def output():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file part"
        
        file = request.files['image']
        
        if file.filename == '':
            return 'No selected file'
        
        
        file.save(file.filename)
        test_img_path = file.filename
        model = tf.keras.models.load_model('./newtea.hdf5', compile=False)

        disease=predict(test_img_path,model)

        with open(test_img_path, "rb") as img_file:
        # Read the image file and encode it as Base64
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')

        context = {
            'test_img': encoded_image,
            'disease': disease
        }
        return render_template('output.html', **context)

@app.route('/')
def hello():
    return send_file('templates\\index.html')


def predict(test_img_path,model):
    test_image = load_img(test_img_path, target_size = (180,180)) # load image

    test_image = img_to_array(test_image)#/255 # convert image to np array and normalize
    test_image = np.expand_dims(test_image, axis = 0)# change dimention 3D to 4D

    result = model.predict(test_image) # predict diseased palnt or not


    pred = np.argmax(result, axis=1)
    pred = pred[0]
    
    expression = ['Anthracnose','algal leaf', 'bird eye spot', 'brown blight', 'gray light', 'healthy', 'red leaf spot','white spot']

    test_image = plt.imread(test_img_path)
    disease =  expression[pred]
    # plt.imshow(test_image)
 
    return disease

# def predict(image_file, model):
#     # Load the image from the file object
#     test_image = Image.open(image_file)
#     test_image = test_image.resize((180, 180))  # Resize the image if needed

#     # Convert image to numpy array and normalize
#     test_image = np.array(test_image) / 255.0
#     test_image = np.expand_dims(test_image, axis=0)  # Change dimension from 3D to 4D

#     # Predict disease
#     result = model.predict(test_image)
#     pred = np.argmax(result, axis=1)
#     pred = pred[0]

#     disease = ['Anthracnose', 'algal leaf', 'bird eye spot', 'brown blight', 'gray light', 'healthy', 'red leaf spot', 'white spot']

#     return disease[pred]

if __name__ == '__main__':
    app.run(debug=True)
