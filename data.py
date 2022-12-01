import cv2
import os
import numpy as np

camera = cv2.VideoCapture(0)
_, frame = camera.read()

Labels = ['Background', 'Blue', 'Red', 'Yellow', 'Green']

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

    while count < 100:
        status, frame  = camera.read()

        if not status:
            print('Frame not captured')
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_blue = np.array([110,50,50])
        upper_blue = np.array([130,255,255])
        
        lower_red = np.array([161, 155, 84])
        upper_red = np.array([179,255,255])
        
        lower_green = np.array([25, 52, 72])
        upper_green = np.array([102, 255, 255])
        
        lower_yellow = np.array([22, 93, 0])
        upper_yellow = np.array([45, 255, 255])

        b_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        r_mask = cv2.inRange(hsv, lower_red, upper_red)
        g_mask = cv2.inRange(hsv, lower_green, upper_green)
        y_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        result = b_mask + r_mask + g_mask + y_mask

        final = cv2.bitwise_and(frame,frame, mask= result)
##        blue = cv2.bitwise_and(frame,frame, mask= b_mask)
##        green = cv2.bitwise_and(frame, frame, mask= g_mask)
##        yellow = cv2.bitwise_and(frame, frame, mask= y_mask)
        
        cv2.imshow('Video Window', final)
        hsv = cv2.resize(result, (1000,1000))
        
        #cv2.imwrite('C:/Users/Mikhaela Rain Roy/Desktop/UNO/' + folder +'/img' + str(count)+ ' .jpg', final)
        #count = count + 1
        if cv2.waitKey(1) == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()
