import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt

video = cv2.VideoCapture(0)

path = ('C:/Users/Mikhaela Rain Roy/Desktop/UNO/Training/image.jpg')
#frame = video.read()
while video.isOpened():
    rval, frame = video.read()
    cv2.imshow('stream',frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
video.release()

save  = cv2.imwrite('C:/Users/Mikhaela Rain Roy/Desktop/UNO/Training/image.jpg',frame)
image =cv2.imread('C:/Users/Mikhaela Rain Roy/Desktop/UNO/Training/image.jpg')

scale = 60
width = int(image.shape[1]*scale/100)
height =int(image.shape[0]*scale/100)
dimensions = (width, height)
resize_img = cv2.resize(image, dimensions)

thr_value, img_thresh = cv2.threshold(resize_img, 150, 255, cv2.THRESH_BINARY)
img_canny = cv2.Canny(img_thresh, 50, 100)    # standard canny edge detector
contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) #contours is not an image, is a chain of pixel locations

cv2.imshow('Thresh', img_thresh)


    
##    contours = sorted(contours, key = cv2.contourArea, reverse = False)
##
##    for i, c in enumerate(contours):         # loop through all the found contours
##        print(i, ':', hierarchy[0, i])          # display contour hierarchy
##        print('length: ', len(c))               # display numbr of points in contour c
##        perimeter = cv2.arcLength(c, True)     # perimeter of contour c (curved length)
##        area = cv2.contourArea(c)
##        print('perimeter: ', perimeter)
##        print('area: ', area)
##        
##        cv2.drawContours(file, [c], 0, (0, 255, 255), 1)   # paint contour c
##        cv2.putText(file, str(i), (c[0, 0, 0]+20, c[0, 0, 1]+30), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,255))
##        
##        contours = sorted(contours, key = cv2.contourArea, reverse = False)
##        x,y,w,h = cv2.boundingRect(c)
##        crop = img_thresh[y-10:y+h+5, x-20:x+w+30]
##
##        if perimeter > 400 and perimeter < 200:
##            continue
##        image_name = "output_shape_number_" + str(i+1) + ".jpg"
##        cv2.imwrite(image_name, crop)
##        readimage = cv2.imread(image_name)
##        
##        cv2.imshow('Image', img_thresh)
##
##    cv2.waitKey(0)
##    cv2.destroyAllWindows()
