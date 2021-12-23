import requests
from io import BytesIO
import cv2
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from flask import Flask, request, Response
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import tensorflow as tf


# creating a Flask app
application = app = Flask(__name__)

# loading the model
vgg19 = tf.keras.applications.VGG19(
    include_top=True,
    weights="imagenet",
    pooling="max")

# use only fc2 layer for similarity calculation 
basemodel = tf.keras.models.Model(inputs=vgg19.input, outputs=vgg19.get_layer('fc2').output)   # fc2 layer

def get_feature_vector(img):
    """Get the feature vector of an image

    Args:  
        img (PIL.Image): PIL Image object

    Returns: 
        vector (numpy.ndarray): feature vector of the image
    """
    img = np.array(img) 
    img1 = cv2.resize(img, (224, 224)) 
    feature_vector = basemodel.predict(img1.reshape(1, 224, 224, 3)) 
    return feature_vector 

def calculate_similarity(vector1, vector2):
    """Calculate the similarity between two images

    Args:
        vector1 (numpy.ndarray): feature vector of the first image
        vector2 (numpy.ndarray): feature vector of the second image

    Returns:
        float: similarity between the two images
    """
    return cosine_similarity(vector1, vector2).reshape((-1,))[0] 


@app.route('/', methods = ['POST'])
def compare_images():
    if "type" in request.form:
        if int(request.form.get('type')) == 0:
            try: 
                img1 = requests.get(request.form.get('img1')) 
                img2 = requests.get(request.form.get('img2'))
            except Exception:
                return Response('Please provide both input image URLs', 400)

            img1 = Image.open(BytesIO(img1.content)) # convert the image to PIL format
            img2 = Image.open(BytesIO(img2.content)) # convert the image to PIL format

        elif int(request.form.get('type')) == 1:
            try:
                img1=  Image.open(request.files['img1'])
                img2 = Image.open(request.files['img2'])
            except Exception:
                return Response('Please provide both input images', 400) # bad request
        else:
            return Response('Invalid input for field "type"', 400) # bad request
    else:
        return Response('Missing field "type"', 400) # bad request


    f1 = get_feature_vector(img1) # get the feature vector of the first image
    f2 = get_feature_vector(img2) # get the feature vector of the second image

    return {"score": round(100*calculate_similarity(f1, f2),3)}, 200 

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)