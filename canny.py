from venv import create
import cv2
import numpy as np
import count as ct
import overlay as ov
import resieze as rs
from PIL import Image
import crop_pil as crop

# Canny Creation
def create_canny(path):

    folder = "./cropped_cords/" + path

    # grayscale
    img = cv2.imread(folder, cv2.IMREAD_GRAYSCALE)
    hh, ww = img.shape[:2]

    # threshold defualt 128
    # lower more filtering
    # 110 for good camera
    threshold = 27

    thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1]

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

    # print(big_contour[0:10])

    # save 
    write = './measure_canny/images/result2.jpg'

    cv2.imwrite(write, result)
    # cv2.imshow('result', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return write

# length of i and overlay cords
cords = open("./XY_cords.txt", "r")

cords_list = []

for line in cords:

  stripped_line = line.strip()
  line_list = stripped_line.split()
  cords_list.append(line_list)

cords.close()

# entire loop
for i in range(len(cords_list)):
   
   canny = create_canny("resize"+str(i)+".png")

   overlay = ov.overlay_canny("resize"+str(i)+".png", 15)

   resize = rs.resize()

   cords = open("./XY_cords.txt", "r")

   img1 = Image.open(r"./test2.jpg")

   small_path = r"./cropped_cords/final_test.png"

   img2 = Image.open(small_path)

   x_offset=round(int(cords_list[i][0])*1)
   y_offset=round(int(cords_list[i][1])*1)

   img1.paste(img2, (x_offset,y_offset))

   img1.save("./test2.jpg")





