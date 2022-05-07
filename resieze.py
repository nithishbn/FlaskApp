# Import packages
import cv2
import numpy as np
# load the original input image and display it on our screen


def resizeN(path, n):
    folder_path = "./cropped_cords/"
    image = cv2.imread(folder_path + path)
    # cv2.imshow("Original", image)
    dem = image.shape

    # demensions 
    r = image.shape[1] * 10
    a = image.shape[0] * 10
    dim = (int(r), int(a))

    # perform the actual resizing of the image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    write = './cropped_cords/resize'+str(n)+'.png'
    cv2.imwrite(write, resized)
    # cv2.imshow("Resized (Width)", resized)
    # cv2.imshow("Normal", image)
    # cv2.waitKey(0)
    return write, dem

def desize(path, n):
    folder_path = "./cropped_cords/"
    image = cv2.imread(folder_path + path)
    # cv2.imshow("Original", image)
    dem = image.shape
   
    # demensions 
    r = image.shape[1] / 10
    a = image.shape[0] / 10
    dim = (int(r), int(a))

    # perform the actual resizing of the image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    write = './cropped_cords/desize'+str(n)+'.png'
    cv2.imwrite(write, resized)
    # cv2.imshow("Resized (Width)", resized)
    # cv2.imshow("Normal", image)
    # cv2.waitKey(0)
    return write, dem





# def desize(path, n):
#     folder_path = "/Users/owardlaw/Desktop/Measure_hair/measure_canny/images/"
#     image = cv2.imread(folder_path + path)
#     first = resize("cropped0.png")[1][0]
#     second = resize("cropped0.png")[1][1]
#     dim = (int(first), int(second))
#     # perform the actual resizing of the image
#     resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
#     write = '/Users/owardlaw/Desktop/Measure_hair/cropped_cords/desize'+str(n)+'.png'
#     cv2.imwrite(write, resized)
#     # cv2.imshow("Resized (Width)", resized)
#     # cv2.imshow("Normal", image)
#     # cv2.waitKey(0)

#     # return write

def resize():

    folder_path = "./measure_canny/images/overlay.png"
    image = cv2.imread(folder_path)
    dem = image.shape
    
    # demensions 
    r = image.shape[1] / 10
    a = image.shape[0] / 10
    dim = (int(r), int(a))

    # perform the actual resizing of the image
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    write = './cropped_cords/final_test.png'
    cv2.imwrite(write, resized)
    # cv2.imshow('result', resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return write
