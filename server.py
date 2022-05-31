import base64
import codecs
import io
from cv2 import imencode
from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2
import sys
import os
import main as mn
import shutil
from PIL import Image
from urllib.parse import unquote
# Initialize the Flask application
app = Flask(__name__)

# route http posts to this method
@app.route('/api/test', methods=['POST'])
def test():
    r = request
    
    parse = r.data
    # print(parse[:50])
    strOne = parse.partition(b'=')[2]
    strOne = unquote(strOne.decode("utf8")).encode("utf8")
    img = Image.open(io.BytesIO(codecs.decode(strOne.strip(),'base64')))
    img.save("testinput.jpg")

    # one click
    mn.runDetection()
    # mn.crop_source_img("testinput.jpg", "testinput.txt")
    mn.hair_canny("testinput.jpg","testinput.txt",25)
    mn.overlayAfterEdgeDetection()
    # send back to client
    imgRet = cv2.imread("det-input-final.jpg")
    # frame = analyze_and_modify_frame( frame )  
    _, frame  = imencode('.jpg', imgRet)
    
    urlSafeEncodedBytes = base64.b64encode(frame)
    urlSafeEncodedStr = str(urlSafeEncodedBytes, "utf8")
    # print(urlSafeEncodedStr[:50])
    return Response(response=urlSafeEncodedStr, status=200, mimetype="application/json") 
@app.after_request
def add_header(response):
    response.cache_control.max_age = 1
    return response


# from gevent.pywsgi import WSGIServer
# #from yourapplication import app

# http_server = WSGIServer(('', 1234), app)
# http_server.serve_forever()
# # start flask app
app.run(host="0.0.0.0", port=1234)
