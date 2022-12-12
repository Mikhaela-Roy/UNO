import cv2
import os
import numpy as np

camera = cv2.VideoCapture(0)
_, frame = camera.read()

Labels = ['Blue', 'Red', 'Yellow', 'Green']

for label in Labels:
    if not os.path.exists(label):
        os.mkdir(label)

for folder in Labels:
    count = 0
    print('Press s to start data collection for: ' + folder)
    userinput = input()

    if userinput != 's':
        print('Wrong input')
        exit()

    while count < 5:
        status, frame  = camera.read()

        if not status:
            print('Frame not captured')
            break
        
        cv2.imshow('Video Window', frame)
        hsv = cv2.resize(frame, (1000,1000))
        
        cv2.imwrite('C:/Users/Mikhaela Rain Roy/Desktop/UNO/' + folder +'/img' + str(count)+ ' .jpg', frame)
        count = count + 1
        if cv2.waitKey(1) == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()
##    height, width, _ = frame.shape
##
##    cx = int(width/2)
##    cy = int(height/2)
##
##    pixel_center = img2[cy,cx]
##    hue_value = pixel_center[0]
##    
##    color = 'Undefined'
##
##    if hue_value < 5:
##        color = 'Red'
##    elif hue_value < 33:
##        color = 'Yellow'
##    elif hue_value < 78:
##        color = 'Green'
##    elif hue_value < 131:
##        color = 'Blue'
##    else:
##        color = 'Undefined'
##    #red needs to be fixed
##
##    kernel = np.ones((5,5),np.uint8)
##
##    #img_dilate = cv2.dilate(frame,kernel, iterations = 1)
##    img_canny = cv2.Canny(frame, 100,200)
##    contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
##      
##
##    contours = sorted(contours, key = cv2.contourArea, reverse = False)
##    
##    for i, c in enumerate(contours):
##        perimeter = cv2.arcLength(c,True)
##        epsilon = 0.02*perimeter
##        vertex_approx = len(cv2.approxPolyDP(c,epsilon,True))
##        
##        #cv2.putText(frame, str(i), (c[0,0,0]+20,c[0,0,1]+30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
##        [x,y,w,h] = cv2.boundingRect(c)
##        area = cv2.contourArea(c)
##
##        if area > 10000 or area < 2000:
##            continue
##        
##        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
##        #cv2.putText(frame, str(i), (c[0,0,0]+20,c[0,0,1]+30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
##        #cv2.drawContours(frame, contours ,i ,(0,255,0),3)
##
##    pixel_center_bgr = frame[cy,cx]
##    b,g,r = int(pixel_center_bgr[0]),int(pixel_center_bgr[1]), int(pixel_center_bgr[2])
##
##    cv2.putText(frame, color, (cx-200,100),0,1,(b,g,r),2)
##
##    cv2.circle(frame,(cx,cy),5,(25,25,25),3)
##    cv2.imshow('Frame', frame)
##
##    key = cv2.waitKey(1)
##    if key == 27:
##        break
