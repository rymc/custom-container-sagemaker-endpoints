# This is the file that implements a flask server to do inference. You can modify it with your own custom code. 
# It is based on https://github.com/aws-samples/amazon-sagemaker-custom-container/
from __future__ import print_function

import os
import torch
import torchvision.transforms.functional as TF
from PIL import Image

import flask
from flask import jsonify
from net import Net

MODEL_PATH = '/opt/ml/model/'
DATA_PATH = '/tmp/data'

IMG_FOR_INFERENCE = os.path.join(DATA_PATH, 'image_for_inference.jpg')


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

            
class ClassificationService(object):

    @classmethod
    def get_model(cls):
        """Get the model object for this instance.  TODO: don't hardcode the model artfiact filename in here"""
        net = Net() 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.load_state_dict(torch.load(f"{MODEL_PATH}cifar_net.pth", map_location=device))
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

    write_test_image(flask.request.stream) 
    
    image = Image.open(IMG_FOR_INFERENCE)
    x = TF.to_tensor(image)
    x.unsqueeze_(0)

    predictions = ClassificationService.predict(x)

    return_value = { "predictions": {} }
    return_value["predictions"]["class"] = str(predictions[0])
    print(return_value)

    return jsonify(return_value) 