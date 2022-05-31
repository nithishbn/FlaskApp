# import sys
# sys.path.insert(0,"measure_canny")
# from measure_canny.cv_hair import cv_detect
import glob
import os
import cv2
from matplotlib.pyplot import getp
from yolov5 import detect
from PIL import Image
import numpy as np
import count as ct
import overlay as ov
# import server.server as sv
import resieze as rs
from yolov5.utils.plots import Annotator, Colors
from PIL import Image
from numpy import asarray
import cv2
import numpy as np

def streamMicroscope():
    vid = cv2.VideoCapture(1)
    while(True):
        # Capture the video frame 
        # by frame
        ret, frame = vid.read()
    
        # Display the resulting frame
        cv2.imshow('Input', frame)
        key = cv2.waitKey(1)

        if key == 32: #spacebar

            out = cv2.imwrite('testinput.jpg',frame)
            break
    if out:
        det = runDetection()
        # if det:
        #     cv2.imshow("Detection",cv2.imread("testinput.jpg"))
        #     cv2.waitKey(0)
    cv2.destroyAllWindows()
def runDetection():
    detect.run(weights= "./best.pt",source="./testinput.jpg",save_txt=True,project="./",name="",hide_labels=True,conf_thres=0.01,save_conf=True)
def overlayAfterEdgeDetection():
    colors = Colors()
    im0 = Image.open("det-input.jpg")
    # np.ascontiguousarray()
    dimensions = im0.size
    dw=dimensions[0]
    dh=dimensions[1]
    im0 = np.ascontiguousarray(im0)
    annotator = Annotator(im0, line_width=1, example='follicle')
    with open("testinput.txt","r") as readFile:
        for line in readFile:
            line = line.split(" ")
            # xyxy = line[1:len(line)]
            x1,y1,x2,y2 = yolobbox2bbox(dw,dh,float(line[1]),float(line[2]),float(line[3]),float(line[4]))
            xyxy = [x1,y1,x2,y2]
            conf = float(line[5])
            label = f'{conf:.2f}'
            annotator.box_label(xyxy, label, color=colors(1,True))
    im0 = annotator.result()
    # Image.fromarray(cv2.cvtColor(im0, cv2.COLOR_BGR2RGBA)).save("det-input-final.jpg", quality=95, subsampling=0)
    Image.fromarray(im0).save("det-input-final.jpg", quality=95, subsampling=0)
    # cv2.imwrite("det-input-final.jpg", im0)
# streamMicroscope()
# cv_detect()
# runDetection()
# print(getPath())

def yolobbox2bbox(dw,dh,x,y,w,h):
    x1 = int((x - w / 2) * dw)
    x2 = int((x + w / 2) * dw)
    y1 = int((y - h / 2) * dh)
    y2 = int((y + h / 2) * dh)

    if x1 < 0:
        x1 = 0
    if x2 > dw - 1:
        x2 = dw - 1
    if y1 < 0:
        y1 = 0
    if y2 > dh - 1:
        y2 = dh - 1

    return x1, y1, x2, y2



def hair_canny(img_path, txt_path, canny_value):

    with open('XY_cords.txt', 'w') as f:
            f.writelines("")

    # start image here
    img = Image.open("./"+img_path)
    x_cord =[]
    y_cord =[]

    # YoloV5 Txt
    cords = open("./"+txt_path, "r")
    cords_list = []
    for line in cords:

        stripped_line = line.strip()
        line_list = stripped_line.split()
        cords_list.append(line_list)

    cords.close()
    x_val = []
    y_val = []
    w_val = []
    h_val = []

    for i in range(len(cords_list)):

        x = float(cords_list[i][1])
        y = float(cords_list[i][2])
        w = float(cords_list[i][3])
        h = float(cords_list[i][4])

        x_val.append(x)
        y_val.append(y)
        w_val.append(w)
        h_val.append(h)

    imgs = []

    for i in range(len(x_val)):
        # print size 
        dimensions = img.size

        dw=dimensions[0]
        dh=dimensions[1]

        x=x_val[i]
        y=y_val[i]
        w=w_val[i]
        h=h_val[i]

        def yolobbox2bbox(x,y,w,h):
            x1 = int((x - w / 2) * dw)
            x2 = int((x + w / 2) * dw)
            y1 = int((y - h / 2) * dh)
            y2 = int((y + h / 2) * dh)

            if x1 < 0:
                x1 = 0
            if x2 > dw - 1:
                x2 = dw - 1
            if y1 < 0:
                y1 = 0
            if y2 > dh - 1:
                y2 = dh - 1

            return x1, y1, x2, y2

        x1 = yolobbox2bbox(x,y,w,h)[0]
        y1 = yolobbox2bbox(x,y,w,h)[1]
        x2 = yolobbox2bbox(x,y,w,h)[2]
        y2 = yolobbox2bbox(x,y,w,h)[3]

        x_cord.append(x1)
        y_cord.append(y1)

        img2 = img.crop((x1, y1, x2, y2))

        data = asarray(img2)

        imgs.append(data)

    # xy list creation
    xy_list = []
    for i in range(len(imgs)):
       xy_list.append([x_cord[i],  y_cord[i]])

    
    # resize
    resized_imgs = []
    for i in range(len(imgs)):

        image = imgs[i]
        dem = imgs[i].shape

        # demensions 
        r = image.shape[1] * 10
        a = image.shape[0] * 10
        dim = (int(r), int(a))

        # perform the actual resizing of the image
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        resized_imgs.append(resized)

    # canny on resized 

    canny = []

    for i in range(len(imgs)):

        # canny_img = Image.fromarray(resized_imgs[i])

        canny_img = resized_imgs[i]


        img = cv2.cvtColor(canny_img, cv2.COLOR_BGR2GRAY)
        hh, ww = img.shape[:2]

        # threshold defualt 128, lower more filtering, 110 for good camera
        thresh = cv2.threshold(img, canny_value, 255, cv2.THRESH_BINARY)[1]

        # get contour
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        big_contour = max(contours, key=cv2.contourArea)

        # draw USE BITWISE 
        result = np.zeros_like(img)
        result = cv2.bitwise_not(result)

        b = (255,255,255)
        w = (0,0,0)

        cv2.drawContours(result, [big_contour], 0, w, cv2.FILLED)

        # save 
        data = asarray(result)

        canny.append(data)

    # overlay
    overlay = []

    for i in range(len(imgs)):

        image = canny[i]
        img_display = resized_imgs[i]

        # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # apply binary thresholding
        ret, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)

        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    
        # counting the number of pixels
        number_of_white_pix = np.sum(image == 255)
        number_of_black_pix = np.sum(image == 0)

        sum = number_of_white_pix + number_of_black_pix
        
        perc = round(((number_of_black_pix/sum)*100), 2)

        if perc >= 86:
            color_canny = (255,0,0)


        elif (perc > 82) and (86 > perc):
            color_canny = (255,255,0)

        else:
            color_canny = (0, 255, 0)

        # thickness_value = thickness_value

        image_disp = img_display.copy()

        cv2.drawContours(image=image_disp, contours=contours, contourIdx=-1, color=color_canny, thickness=20, lineType=cv2.LINE_AA)


        write = './measure_canny/images/overlay.png'

        cv2.imwrite(write, image_disp)


        overlay.append(image_disp)

    # desize
    desized_imgs = []
    for i in range(len(imgs)):

        image = overlay[i]
        dem = overlay[i].shape

        # demensions 
        r = image.shape[1] / 10
        a = image.shape[0] / 10
        dim = (int(r), int(a))

        # perform the actual resizing of the image
        desized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        desized_imgs.append(desized)

    # overlay back 
    img1 = Image.open(r"./det-input.jpg")
    for i in range(len(imgs)):

        

        # Change path for BOX overlay
        small_path = Image.fromarray(desized_imgs[i])

        x_offset=round(int(xy_list[i][0]))
        y_offset=round(int(xy_list[i][1]))

        img1.paste(small_path, (x_offset,y_offset))

    img1.save("./det-input.jpg")


    return imgs, xy_list, resized_imgs, canny, overlay, desized_imgs


