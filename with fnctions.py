import cv2
import numpy as np
import glob

path = glob.glob('C:/Users/Mikhaela Rain Roy/Desktop/UNO/UNO images/*.jpg')

def crop(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr_value, img_thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    img_canny = cv2.Canny(img_thresh, 50, 100)    # standard canny edge detector
    contours, hierarchy = cv2.findContours(img_thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #contours is not an image, is a chain of pixel locations
    kernel = np.ones((3,3),np.float32)/25
    img_close = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel)
    dilation = cv2.dilate(img_close, kernel,iterations = 1)
    erosion = cv2.erode(dilation, kernel, iterations = 1)
    dst = cv2.filter2D(erosion,-1,kernel)
        
    for i, c in enumerate(contours):         # loop through all the found contours
        
        print(i, ':', hierarchy[0, i])          # display contour hierarchy
        print('length: ', len(c))               # display numbr of points in contour c
        perimeter = cv2.arcLength(c, True)     # perimeter of contour c (curved length)
        area = cv2.contourArea(c)
        print('perimeter: ', perimeter)
        print('area: ', area)
        
        cv2.drawContours(img, [c], 0, (0, 255, 255), 1)   # paint contour c
        #cv2.putText(img, str(i), (c[0, 0, 0]+20, c[0, 0, 1]+30), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,255))

        contours = sorted(contours, key = cv2.contourArea, reverse = False)
        
        x,y,w,h = cv2.boundingRect(c)
        cropped = img_canny[y:y+h, x:x+w]

        if perimeter >= 400 or perimeter <=200:
            continue

        cv2.imshow('Image', cropped)

        image_name = "output_shape_number_" + str(i+1) + ".jpg"
        cv2.imwrite(image_name, cropped)
        readimage = cv2.imread(image_name)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

for image in path:
    img = cv2.imread(image)
    
    crop(img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    red_lower = np.array([136, 87, 111])
    red_upper = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
  
    # Set range for green color and 
    # define mask
    green_lower = np.array([25, 52, 72])
    green_upper = np.array([102, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
  
    # Set range for blue color and
    # define mask
    blue_lower = np.array([94, 80, 2])
    blue_upper = np.array([120, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    result = red_mask + green_mask + blue_mask
    output = cv2.bitwise_and(img, img, mask = result)

    cv2.imshow('Colour Detected', np.hstack((img,output)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
