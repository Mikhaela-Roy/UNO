import cv2

img = cv2.imread('./Training/b1.jpg')

scale = 200

width = int(img.shape[1]*scale/100)
height = int(img.shape[0]*scale/100)

dsize = (width, height)
out = cv2.resize(img,dsize)

img_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
thr_value, img_thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
img_canny = cv2.Canny(img_thresh, 80, 100)    # standard canny edge detector
contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #contours is not an image, is a chain of pixel locations

#contours = sorted(contours, key = cv2.contourArea)

for i, c in enumerate(contours):         # loop through all the found contours
    print(i, ':', hierarchy[0, i])          # display contour hierarchy
    print('length: ', len(c))               # display numbr of points in contour c
    perimeter = cv2.arcLength(c, True)     # perimeter of contour c (curved length)
    area = cv2.contourArea(c)
    print('perimeter: ', perimeter)
    print('area: ', area)
    
    cv2.drawContours(out, [c], 0, (0, 255, 0), 1)
    cv2.putText(out, str(i), (c[0, 0, 0]+20, c[0, 0, 1]+30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
    
    x,y,w,h = cv2.boundingRect(c)
    crop = out[y:y+h, x:x+w]

cv2.imshow('Crop contour', out) 
cv2.waitKey(0)
cv2.destroyAllWindows()
