from cv2 import imencode
from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2
import sys
import os
import main as mn
import shutil

# Initialize the Flask application
app = Flask(__name__)

# route http posts to this method
@app.route('/api/test', methods=['POST'])
def test():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite("testinput.jpg", img)
    # if os.path.exists("det-input.jpg"):
    #     os.remove("det-input.jpg")
    # if os.path.exists("testinput.txt"):
    #     os.remove("testinput.txt")
    # one click
    mn.runDetection()
    # mn.crop_source_img("testinput.jpg", "testinput.txt")
    mn.hair_canny("testinput.jpg","testinput.txt",25)
    mn.overlayAfterEdgeDetection()
    # send back to client
    imgRet = cv2.imread("det-input-final.png")
    # do some fancy processing here....
    # frame = analyze_and_modify_frame( frame )  
    _, frame  = imencode('.png', imgRet)
    response_pickled  = jsonpickle.encode( frame )
    return Response(response=response_pickled, status=200, mimetype="application/json") 
@app.after_request
def add_header(response):
    response.cache_control.max_age = 300
    return response

# start flask app
app.run(host="0.0.0.0", port=1234)