from __future__ import print_function
import requests
import json
import cv2
import jsonpickle

from main import streamMicroscope
addr = 'http://76.176.144.205:1234'
# addr = "http://localhost:1234/"
test_url = addr + '/api/test'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}
streamMicroscope()
img = cv2.imread('testinput.jpg')
# img = cv2.imread('photo_2022-05-06_19-58-00.jpg')
# encode image as jpeg
_, img_encoded = cv2.imencode('.png', img)
# print(img_encoded.tostring())
# send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
# decode response
# print(json.loads(response.text))
frame = jsonpickle.decode(response.text) 
frame = frame.tobytes()
with open("fromserver-final.png","wb") as writeFile:
    writeFile.write(frame)

# expected output: {u'message': u'image received. size=124x124'}