from flask import Flask, jsonify, request, send_file
import os
import json
import numpy as np
import time
import io
from get_binvox import *
import cv2

app = Flask(__name__, static_url_path='', static_folder='client/build')

@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route('/generateBinvox', methods=["GET", "POST"])
def endAnalysis():
    if request.method == "GET":
        with open("model.binvox", 'rb') as bites:
            return send_file(
                        io.BytesIO(bites.read()),
                        attachment_filename='model.binvox',
                        as_attachment=True,
                        mimetype='application/octet-stream'
                )

    elif request.method == "POST":
        try:
            print('asdklfdklsj')
            print(request.files['0'])
            request.files['0'].save('0.jpg')
            request.files['1'].save('1.jpg')
            request.files['2'].save('2.jpg')
            request.files['3'].save('3.jpg')
            try:
                main()
            except Exception as e:
                print(e)
            return "200"
            
        except Exception as e:
            return str(e)

if __name__ == '__main__':
    #app.run()
    app.run(host='0.0.0.0', port=80, debug=True)
