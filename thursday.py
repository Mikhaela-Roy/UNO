import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt

def image():
    path = 'C:/Users/Mikhaela Rain Roy/Desktop/UNO/Training/*.jpg'
    image = [cv2.imread(image) for image in glob.glob(path)]

    for i in image:
        scale = 500

        width = int(i.shape[1]*scale/100)
        height = int(i.shape[0]*scale/100)

        dsize = (width, height)
        out = cv2.resize(i,dsize)
        
    return i, out

i, out= image()

for g in i:
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
    
cv2.imshow('2D Convultion', dst)
cv2.imshow('Erosion', erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
