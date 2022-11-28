import cv2
import numpy as np

img = cv2.imread('./UNO images/b0.jpg') # img = cv2.imread('./images/r5.jpg') # 
##cv2.imshow('image',img)

cropped_img = img[250:450, 200:450]

#cv2.imshow("Original image", img)
#cv2.imshow("Cropped image", cropped_img)

def convolution(image, kernel):
    
    k = int((len(kernel)-1)/2)   # k is the kernal radius, used to generalise and simplify equations
    new_img = np.zeros((len(image),len(img[0])), dtype=np.uint8) # make sure you don't overwrite the original image

    for r in range(k, len(img)-k): # loop through all rows, omitting the first and last k
        for c in range(k, len(image[0])-k): # loop through all columns, omitting the first and last k
            patch = np.array(image[r-k:r+k+1, c-k:c+k+1]) # extract an image patch the same size of the kernel, around the target pixel
            new_img[r][c] = sum(sum(patch * kernel)) # the target pixel take the value of the weighted average of its neighbour, computed as sum of products of patch and kernel

    return new_img

kernel_size = 3
smoothing_kernel = np.ones((kernel_size, kernel_size))/kernel_size**2 # [[1,1,1],[1,1,1],[1,1,1]]/9 or equivalent for bigger kernels
img_smooth = convolution(img, smoothing_kernel)    # apply convolution with the desired kernel

cv2.imshow('smoothing3', img_smooth)

img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
thr_value, img_thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
img_canny = cv2.Canny(img_thresh, 50, 100)    # standard canny edge detector
contours, hierarchy = cv2.findContours(img_thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #contours is not an image, is a chain of pixel locations

cv2.drawContours(cropped_img, contours,-1,(0,0,255),1) # paint contours on top of original image
cv2.imshow("colour_img", cropped_img)# display each frame as an image, "stream" is the name of the window

for i, c in enumerate(contours):         # loop through all the found contours
    print(i, ':', hierarchy[0, i])          # display contour hierarchy
    print('length: ', len(c))               # display numbr of points in contour c
    perimeter = cv2.arcLength(c, True)     # perimeter of contour c (curved length)
    print('perimeter: ', perimeter)
    
    cv2.drawContours(cropped_img, [c], 0, (0, 255, 0), 3)   # paint contour c
    cv2.putText(cropped_img, str(i), (c[0, 0, 0]+20, c[0, 0, 1]+30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))
    [x,y,w,h] = cv2.boundingRect(c)
    cv2.rectangle(cropped_img, (x,y), (x+w,y+h), (255, 0, 0), 2)
    
cv2.namedWindow('picture', cv2.WINDOW_NORMAL)
cv2.imshow('picture',cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

