import io
import json

import flask
from flask import Flask, render_template, request, jsonify, make_response
import numpy as np
import torch

import torchvision.transforms as transforms
from PIL import Image


app = Flask(__name__)
model = torch.load("model5.pth")

#transform and normalize for densenet standard
def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    model = torch.load("model5.pth")
    imagenet_class_index = json.load(open('class_index.json')) 
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


@app.route("/")
def home():
    return flask.render_template('home.html')

@app.route("/index")
def index():
   return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method=='POST':
         file = request.files['image']
         if not file: 
            return render_template('index.html', label="No file")
         img_bytes = file.read()
         class_id, class_name = get_prediction(image_bytes=img_bytes)
         final_prediction = jsonify({'class_name': class_name})
         return render_template('index.html', label = class_name)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
    
