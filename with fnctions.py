import cv2
import numpy as np
import glob

def AllImage():
    path = glob.glob('C:/Users/Mikhaela Rain Roy/Desktop/UNO/UNO images/*.jpg')

    for image in path:
        img = cv2.imread(image)
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thr_value, img_thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
        img_canny = cv2.Canny(img_thresh, 50, 100)    # standard canny edge detector
        contours, hierarchy = cv2.findContours(img_thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #contours is not an image, is a chain of pixel locations
        kernel = np.ones((5,5),np.float32)/25
        img_close = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)
        thresh4 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 4)
        dilation = cv2.dilate(thresh4, kernel,iterations = 1)
        erosion = cv2.erode(dilation, kernel, iterations = 1)
        dst = cv2.filter2D(erosion,-1,kernel)

        cv2.imshow('Image', dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
AllImage()
    

