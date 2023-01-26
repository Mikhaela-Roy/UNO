import numpy as np
import cv2 
import glob
import os

vc = cv2.VideoCapture(0)

#https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
def compare():
    
    img1 = cv2.imread('./UNO images/b0.jpg')
    orb = cv2.ORB_create()# queryImage
    kp1, des1 = orb.detectAndCompute(img1, None)
    path = glob.glob("./UNO images/*.jpg")
    cv_img = []
    l=0
    
    for img in path:
        img2 = cv2.imread(img) # trainImage
        # Initiate SIFT detector

        # find the keypoints and descriptors with SIFT

        kp2, des2 = orb.detectAndCompute(img2, None)
        # feature matching
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        if(l<len(matches)):
            l=len(matches)
            image=img2
            m=matches

    kp2, des2 = orb.detectAndCompute(image,None)
    cv2.imshow('Match',image)

def num_detect(vc):
    
    while vc.isOpened():

        rval, frame = vc.read()
        copy = frame.copy()

        cv2.imshow('Input',copy)
        
        key = cv2.waitKey(1)
        if key == 27:
            break
        
        #capture last frame
        saved = cv2.imwrite('./UNO/image.jpg', frame)
        
    cv2.destroyAllWindows()
    vc.release()
num_detect(vc)
