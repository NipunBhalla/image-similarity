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

vgg19 = tf.keras.applications.VGG19(
    include_top=True,
    weights="imagenet",
    pooling="max")

basemodel = tf.keras.models.Model(inputs=vgg19.input, outputs=vgg19.get_layer('fc2').output)   


def get_feature_vector(img):
    img = np.array(img)
    img1 = cv2.resize(img, (224, 224))
    feature_vector = basemodel.predict(img1.reshape(1, 224, 224, 3))
    return feature_vector

def calculate_similarity(vector1, vector2):
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

            img1 = Image.open(BytesIO(img1.content))
            img2 = Image.open(BytesIO(img2.content))

        elif int(request.form.get('type')) == 1:
            try:
                img1=  Image.open(request.files['img1'])
                img2 = Image.open(request.files['img2'])
            except Exception:
                return Response('Please provide both input images', 400)
        else:
            return Response('Invalid input for field "type"', 400)
    else:
        return Response('Missing field "type"', 400)


    f1 = get_feature_vector(img1)
    f2 = get_feature_vector(img2)

    return {"score": round(100*calculate_similarity(f1, f2),3)}, 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)