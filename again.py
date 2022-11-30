import cv2
import numpy as np
import glob

#img= cv2.imread('./images/g8.jpg')#bS, bT, bR
path = glob.glob('C:/Users/Olu/desktop/images/*.jpg')

images = [cv2.imread(file) for file in glob.glob('C:/Users/Olu/desktop/images/*.jpg')]

##img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
##thr_value, img_thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
##img_canny = cv2.Canny(img_thresh, 50, 100)    # standard canny edge detector
##contours, hierarchy = cv2.findContours(img_thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #contours is not an image, is a chain of pixel locations
##    
##for i, c in enumerate(contours):         # loop through all the found contours
##    print(i, ':', hierarchy[0, i])          # display contour hierarchy
##    print('length: ', len(c))               # display numbr of points in contour c
##    perimeter = cv2.arcLength(c, True)     # perimeter of contour c (curved length)
##    area = cv2.contourArea(c)
##    print('perimeter: ', perimeter)
##    print('area: ', area)
##    
##    cv2.drawContours(img, [c], -1, (0, 255, 0), 1, cv2.LINE_AA)   # paint contour c
##    #cv2.putText(img, str(i), (c[0, 0, 0]+20, c[0, 0, 1]+30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
##    
##    contours = sorted(contours, key=cv2.contourArea, reverse = True)
##
##    x,y,w,h = cv2.boundingRect(c)
##    crop = img[y-10:y+h+5, x-20:x+w+30]
##
##    image_name= "output_shape_number_" + str(i+1) + ".jpg"
##    cv2.imwrite(image_name, crop)
##    readimage= cv2.imread(image_name)
##    cv2.imshow('Image', crop)
##    
##cv2.imshow('Original',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

##import pickle
##import numpy as np
##from sklearn import datasets
##from sklearn.model_selection import train_test_split
##from sklearn.neural_network import MLPClassifier
##
##iris = datasets.load_iris()
##X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=.8, random_state=42)
##
##classifier = MLPClassifier()
##classifier.fit(X_train, y_train)
##pickle.dumo(classifier, open("iris_classifier","wb"))
