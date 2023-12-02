# This is the file that implements a flask server to do inferences. It's the file that you will modify
# to implement the prediction for your own algorithm.

from __future__ import print_function

import os, sys, stat
import json
import shutil

import torch
import torchvision.transforms.functional as TF
from PIL import Image

import flask
from flask import Flask, jsonify
import glob
from net import Net


MODEL_PATH = '/opt/ml/model/cifar_net.pth'
DATA_PATH = '/tmp/data'

IMG_FOR_INFERENCE = os.path.join(DATA_PATH, 'image_for_inference.jpg')

# in this tmp folder, image for inference will be saved
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH, mode=0o755,exist_ok=True)


def write_test_image(stream):
    with open(IMG_FOR_INFERENCE, "bw") as f:
        chunk_size = 4096
        while True:
            chunk = stream.read(chunk_size)
            if len(chunk) == 0:
                return
            f.write(chunk)

            
# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
class ClassificationService(object):

    @classmethod
    def get_model(cls):
        """Get the model object for this instance."""
        net = Net()
        net.load_state_dict(torch.load(MODEL_PATH))
        return net

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them."""
        model = cls.get_model()
        outputs = model(input)
        _, predicted = torch.max(outputs, 1)   
        return predicted

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ClassificationService.get_model() is not None  

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():

    write_test_image(flask.request.stream) #receive the image and write it out as a JPEG file.
    
    image = Image.open(IMG_FOR_INFERENCE)
    x = TF.to_tensor(image)
    x.unsqueeze_(0)

    # Do the prediction
    predictions = ClassificationService.predict(x) #predict() also loads the model

    # Convert result to JSON
    return_value = { "predictions": {} }
    return_value["predictions"]["class"] = str(predictions[0])
    print(return_value)

    return jsonify(return_value) 