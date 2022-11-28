import cv2

# Load image, convert to grayscale, and find edges
image = cv2.imread('./UNO images/b2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

# Find contour and sort by contour area
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv2.fitEllipse, reverse=True)
cv2.drawContours(image, cnts,-1,(0,0,255),1) # paint contours on top of original image

# Find bounding box and extract ROI
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    cv2.drawContours(image, [c], 0, (0, 255, 0), 2)   # paint contour c
   # cv2.putText(image, (c[0, 0, 0]+20, c[0, 0, 1]+30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
   # [x,y,w,h] = cv2.boundingRect(c)
   # cv2.rectangle(image, (x,y), (x+w,y+h), (255, 0, 0), 2)

    card = image[y:y+h, x:x+w]
    break

cv2.imshow('UNO',card)
cv2.waitKey(0)
cv2.destroyAllWindows()
