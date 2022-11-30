import cv2

img= cv2.imread('./images/b1.jpg') # img = cv2.imread('./images/r5.jpg') # 
##cv2.imshow('image',img)

#cropped_img = img[250:450, 200:450]

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thr_value, img_thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
img_canny = cv2.Canny(img_thresh, 50, 100)    # standard canny edge detector
contours, hierarchy = cv2.findContours(img_thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #contours is not an image, is a chain of pixel locations

#cv2.drawContours(img, contours,-1,(0,0,255),1) # paint contours on top of original image
#cv2.imshow("colour_img", img)# display each frame as an image, "stream" is the name of the window

for i, c in enumerate(contours):         # loop through all the found contours
    print(i, ':', hierarchy[0, i])          # display contour hierarchy
    print('length: ', len(c))               # display numbr of points in contour c
    perimeter = cv2.arcLength(c, True)     # perimeter of contour c (curved length)
    area = cv2.contourArea(c)
    print('perimeter: ', perimeter)
    print('area: ', area)

##    contours = contours[0] if len(contours) == 2 else contours[1]
##    contours = sorted(contours, key=cv2.boundingRect, reverse=True)
##
##    x,y,w,h = cv2.boundingRect(c)
    
    cv2.drawContours(img, [c], 0, (0, 255, 0), 1)   # paint contour c
    cv2.putText(img, str(i), (c[0, 0, 0]+20, c[0, 0, 1]+30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))

    #crop = img[y:y+h, x:x+w].copy()
    
    
#cv2.namedWindow('picture', cv2.WINDOW_NORMAL)
cv2.imshow('card', img)
#cv2.imshow('cropped',crop)
cv2.waitKey(0)
cv2.destroyAllWindows()
