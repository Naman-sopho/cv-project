from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import os
# import pymysql
# import math
# import numpy as np
# import pandas as pd
# import pymysql.cursors
import json
import lasagne
import theano
import theano.tensor as T
import numpy as np

import glob
#from scipy.misc.pilutil import imread
from scipy.misc import imresize, imshow, imread
import time

import models


def read_model(layer, filename):
    """ Load the weights of a network """
    with np.load(filename) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(layer, param_values)

def imcrop(image, crop_range):
    """ Crop an image to a crop range """
    return image[crop_range[0][0]:crop_range[0][1],
                 crop_range[1][0]:crop_range[1][1], ...]



app = Flask(__name__, static_url_path='', static_folder='client/build')
global ultrasound_loc, image_label, patient_img, done_analysis, confidence
ultrasound_loc = -1
image_label = ""
patient_img = -1
done_analysis = False
confidence = ""

network_name = 'SonoNet32'

# The mapping from network output to label name
label_names = [ '3VV',
                '4CH',
                'Abdominal',
                'Background',
                'Brain (Cb.)',
                'Brain (Tv.)',
                'Femur',
                'Kidneys',
                'Lips',
                'LVOT',
                'Profile',
                'RVOT',
                'Spine (cor.)',
                'Spine (sag.) ']

# Crop range used to get rid of the vendor info etc around the images
crop_range = [(115, 734), (81, 874)]  # [(top, bottom), (left, right)]

# The input images will be resized to this size
input_size = [224, 288]

# Display the images during the prediction
display_images = False

input_var = T.tensor4('inputs')

# Defining the model and reading the paramters
network_builder = getattr(models,network_name)
net = network_builder(input_var, input_size, num_labels=len(label_names))
read_model(net['output'], '%s.npz' % network_name)

# Defining the prediction function
prediction_var = lasagne.layers.get_output(net['output'], deterministic=True)
pred_and_conf_fn = theano.function(
                        [input_var],
                        [T.argmax(prediction_var, axis=1),
                         T.max(prediction_var, axis=1)]
                        )


#def read_model(layer, filename):
#    """ Load the weights of a network """
#    with np.load(filename) as f:
#        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
#        lasagne.layers.set_all_param_values(layer, param_values)

#def imcrop(image, crop_range):
#    """ Crop an image to a crop range """
#    return image[crop_range[0][0]:crop_range[0][1],
#                 crop_range[1][0]:crop_range[1][1], ...]


# DB config
# basedir = os.path.abspath(os.path.dirname(__file__))

# connection = pymysql.connect(host='104.154.198.152',
#                              user='medhacks',
#                              password='rapidmoon667',
#                              db='medhacks')

@app.route('/')
def root():
    print(request.headers)
    return app.send_static_file('index.html')


@app.route('/Splash', methods=["GET"])
def splash():
    global done_analysis, patient_img, ultrasound_loc, confidence, image_label
    done_analysis = False
    ultrasound_loc = -1
    image_label = ""
    patient_img = -1
    done_analysis = False
    confidence = ""
    return "200"

@app.route('/nextStep')
def nextStep():
    # To be added - Status
    return jsonify()

@app.route('/physician', methods=["GET","POST"])
def physician():
    global image_label
    if request.method == "POST":
        content = request.get_json()["image_count"]
        print(content)
        global ultrasound_loc
        ultrasound_loc = content
        filename = "client/public/ultra" + str(ultrasound_loc) + ".png"
        input_list = []
        image = imread(filename)  # read
        image = imcrop(image, crop_range)  # crop
        image = imresize(image, input_size)  # resize
        image = np.mean(image,axis=2)  # convert to gray scale

        # convert to 4D tensor of type float32
        image_data = np.float32(np.reshape(image,
                                          (1,1,image.shape[0], image.shape[1])))

        # normalise images by substracting mean and dividing by standard dev.
        mean = image_data.mean()
        std = image_data.std()
        image_data = np.array(255.0*np.divide(image_data - mean, std),
                              dtype=np.float32)
                              # Note that the 255.0 scale factor is arbitrary
                              # it is necessary because the network was trained
                              # like this, but the same results would have been
                              # achieved without this factor for training.

        input_list.append(image_data)

        total_time = 0  # measures the total time spent predicting in seconds
        print("\nPredictions using " +  network_name + ":")

        for X in input_list:

            start_time = time.time()
            [prediction, confidence] = pred_and_conf_fn(X)  # get the prediction
            total_time += time.time() - start_time

            #true_label = file_name.split('/')[1].split('.')[0]

            # True labels are obtained from file name.
            # print(" - " + label_names[prediction[0]] + " (conf: " + str(confidence[0]) +  ", true label:$
            image_label = label_names[prediction[0]]
            confidence =  str(confidence[0])

        return "200"

    elif request.method == "GET":
        global patient_img
        return json.dumps({"patient_img": patient_img, "img_label": image_label})

@app.route('/patient', methods=["GET","POST"])
def patient():
    if request.method == "GET":
        global ultrasound_loc
        return json.dumps({"ultra_loc":ultrasound_loc})

    elif request.method == "POST":
        global patient_img
        content = request.get_json()["patient_img_count"]
        print(content)
        patient_img = content
        return "200"

@app.route('/doneAnalysis', methods=["GET", "POST"])
def endAnalysis():
    global done_analysis
    if request.method == "GET":

        return json.dumps({"doneAnalysis": done_analysis})

    elif request.method == "POST":
        content = request.get_json()["doneAnalysis"]
        done_analysis = content
        print(done_analysis)
        return "200"

if __name__ == '__main__':
    #app.run()
    app.run(host='0.0.0.0', port=80)
