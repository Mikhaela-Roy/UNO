import cv2
import numpy as np
import os 

path = ('C:/Users/Mikhaela Rain Roy/Desktop/UNO/UNO images')

orb = cv2.ORB_create(nfeatures = 1000)

images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    img = cv2.imread(f'{path}/{cl}',0)
    images.append(img)
    classNames.append(os.path.splitext(cl)[0])


def find(images):
    desList = []
    for img in images:
        kp,des = orb.detectAndCompute(img, None)
        desList.append(des)
    return desList


def findID(img,desList,thresh=15):
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    matchList = []
    finalVal = -1
    try:
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])

            matchList.append(len(good))
    except:
        pass
    if len(matchList) != 0:
        if max(matchList) > thresh:
            finalVal = matchList.index(max(matchList))
    return finalVal
    #print(matchList)
  
desList = find(images)
print(len(desList))
      
vc = cv2.VideoCapture(0)

while True:
    
    success, frame = vc.read()
    
    Original = frame.copy()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    id = findID(img, desList)
    if id != -1:
        cv2.putText(Original, classNames[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),0,1)
    
    cv2.imshow('img', Original)
    cv2.waitKey(1)
cv2.destroyAllWindows()
