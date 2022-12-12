import cv2 
from matplotlib import pyplot as plt
import numpy as np
 
# keep in mind that open CV loads images as BGR not RGB
vc = cv2.VideoCapture(0)

def find(contours, frame):
    
       for i, c in enumerate(contours):
        #print(i, ':', hierarchy[0, i])          # display contour hierarchy
        #print('length: ', len(c))               # display numbr of points in contour c
        perimeter = cv2.arcLength(c, True)     # perimeter of contour c (curved length)
        area = cv2.contourArea(c)
        #print('perimeter: ', perimeter)
        #print('area: ', area)

        x,y,w,h = cv2.boundingRect(c)
        cropped = img_canny[y:y+h, x-20:x+w+20]

        features = []
        
        if perimeter > 800 or perimeter < 850:
            saved = cv2.imwrite('C:/Users/Mikhaela Rain Roy/Desktop/UNO/Training/image.jpg', frame)
##        features.append([area,perimeter])
##        print(features)

def process():
    
    image = imread('C:/Users/Mikhaela Rain Roy/Desktop/UNO/Training/image.jpg')

    scale = 60
    width = int(image.shape[1]*scale/100)
    height = int(image.shape[0]*scale/100)
    dimension = (width,height)

while vc.isOpened():

    rval, frame = vc.read()

    img_canny = cv2.Canny(frame, 100,200)
    contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    contours = sorted(contours, key = cv2.contourArea, reverse = False)
    
    for i, c in enumerate(contours):
        perimeter = cv2.arcLength(c,True)
        epsilon = 0.02*perimeter
        vertex_approx = len(cv2.approxPolyDP(c,epsilon,True))
        
        #cv2.putText(frame, str(i), (c[0,0,0]+20,c[0,0,1]+30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
        [x,y,w,h] = cv2.boundingRect(c)
        areaContour = cv2.contourArea(c)

        if areaContour > 2000 or 100000 < areaContour:
            continue
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
        #cv2.putText(frame, str(i), (c[0,0,0]+20,c[0,0,1]+30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
        #cv2.drawContours(frame, contours ,i ,(0,255,0),3)
        
    cv2.imshow('Image', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
vc.release()
