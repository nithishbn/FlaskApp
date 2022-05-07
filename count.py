import cv2
import numpy as np
  

def thickness_perc(path):

    # reading the image data from desired directory
    img = cv2.imread(path)
    
    # counting the number of pixels
    number_of_white_pix = np.sum(img == 255)
    number_of_black_pix = np.sum(img == 0)

    sum = number_of_white_pix + number_of_black_pix
    
    perc = round(((number_of_black_pix/sum)*100), 2)


    return perc



