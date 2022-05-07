import cv2
import numpy as np
import count as ct


def overlay_canny(path, thickness_value):
    # read the image
    image = cv2.imread("./measure_canny/images/result2.jpg")
    img_display = cv2.imread("./cropped_cords/" + path)

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply binary thresholding
    ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
   
    # draw contours on the original image
    image_copy = image.copy()

    thick = ct.thickness_perc("./measure_canny/images/result2.jpg")

    if thick >= 86:
        color_canny = (0, 0, 255)


    elif (thick > 82) and (86 > thick):
        color_canny = (0, 255, 255)

    else:
        color_canny = (0, 255, 0)

    thickness_value = thickness_value

    image_disp = img_display.copy()

    cv2.drawContours(image=image_disp, contours=contours, contourIdx=-1, color=color_canny, thickness=20, lineType=cv2.LINE_AA)


    write = './measure_canny/images/overlay.png'

    cv2.imwrite(write, image_disp)



    return write


