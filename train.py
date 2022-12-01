import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt

path = 'C:/Users/Mikhaela Rain Roy/Desktop/UNO/Training/*.jpg'
image = [cv2.imread(image) for image in glob.glob(path)]

for i in image:
    scale = 500

    width = int(i.shape[1]*scale/100)
    height = int(i.shape[0]*scale/100)

    dsize = (width, height)
    out = cv2.resize(i,dsize)

    img_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    thr_value, img_thresh = cv2.threshold(img_gray, 90, 255, cv2.THRESH_BINARY)
    img_canny = cv2.Canny(img_thresh, 150, 100)    # standard canny edge detector
    contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #kernel_size = 3
    #kernel = np.ones((kernel_size, kernel_size), np.uint8)
    kernel = np.ones((5,5),np.float32)/25
    img_close = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
    thresh4 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 4)
    dilation = cv2.dilate(thresh4,kernel,iterations = 1)
    erosion = cv2.erode(dilation, kernel, iterations = 1)
    dst = cv2.filter2D(erosion,-1,kernel)


##    for c in enumerate(contours):
##        cv2.drawContours(img_thresh, [c], 0, (0, 255, 0), 1)
##        cv2.putText(out, str(i), (c[0, 0, 0]+20, c[0, 0, 1]+30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))

    for i, c in enumerate(contours):         # loop through all the found contours
        cv2.drawContours(out, [c], 0, (0, 255, 0), 1)
        cv2.putText(out, str(i), (c[0, 0, 0]+20, c[0, 0, 1]+30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
        
    cv2.imshow('2D Convultion', dst)
    cv2.imshow('Erosion', erosion)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#next step, thresholding and then binarisation
##contours is not an image, is a chain of pixel locations
##
##contours = sorted(contours, key = cv2.contourArea)
##resize_contour(im)
##
##for i, c in enumerate(contours):         # loop through all the found contours
##    print(i, ':', hierarchy[0, i])          # display contour hierarchy
##    print('length: ', len(c))               # display numbr of points in contour c
##    perimeter = cv2.arcLength(c, True)     # perimeter of contour c (curved length)
##    area = cv2.contourArea(c)
##    print('perimeter: ', perimeter)
##    print('area: ', area)
##    
##    cv2.drawContours(out, [c], 0, (0, 255, 0), 1)
##    cv2.putText(out, str(i), (c[0, 0, 0]+20, c[0, 0, 1]+30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
##    
##    x,y,w,h = cv2.boundingRect(c)
##    crop = out[y:y+h, x:x+w]
##
##cv2.imshow('Crop contour', out) 
##cv2.waitKey(0)
##cv2.destroyAllWindows()
