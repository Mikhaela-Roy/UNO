import cv2
import numpy as np
import glob


class Query_card:
    
    def __init__(self):
        self.contour = []
        self.width, self.height  = 0,0
        self.corner_pts = []
        self.center = []
        self.warp = []
        self.rank_img = []
        self.colour_img = []
        self.best_rank_match = 'Unknonw'
        self.best_color_match = 'Unknown'
        self.rank_diff = 0
        self.colour_diff = 0

class Train:
    
    def __init__(self):
        self.img = []
        self.name = 'Placeholder'

class Train_colour:

    def __init__(self):
        self.img = []
        self.name = 'Placeholder'

def load_rank(filepath):

    train_rank = []
    i = 0

    for Rank in ['1','2','3','4','5','6','7','8','9']:
        train_rank.append(Train())
        train_rank[i].name = Rank
        filename = Rank +'.jpg'
        train_rank[i].img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)
        i += 1

    return train_rank

def load_colour(filepath):

    train_colour = []
    i = 0

    for Colour in ['blue','yellow','red','green']:
        train_colour.append(Train_colour())
        train_colour[i].name = Colour
        filename = Colour +'.jpg'
        train_colour[i].img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)
        i += 1

    return train_colour
        
def preprocess(image):
    
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gray,(5,5),0)

    img_w, img_h = np.shape(img)[:2]
    bkg_level = img_gray[int(img_h/100)][int(img_w/2)]
    thresh_level = bkg_level + 60

    
    thr_value, img_thresh = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY)

    return thresh

def find(thresh_image):
    
    img_canny = cv2.Canny(img_thresh, 50, 100)    # standard canny edge detector
    contours, hierarchy = cv2.findContours(img_thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #contours is not an image, is a chain of pixel locations
    kernel = np.ones((3,3),np.float32)/25
    img_close = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, kernel)
    dilation = cv2.dilate(img_close, kernel,iterations = 1)
    erosion = cv2.erode(dilation, kernel, iterations = 1)
    dst = cv2.filter2D(erosion,-1,kernel)

    index_sort = sorted(range(len(contours)), key = lambda i: cv2.contourArea(contours[i]), reverse = False)

    if len(contours) == 0:
        return [],[]

    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(contours), dtype= int)

    for i in index_sort:
        cnts_sort.append(contours[i])
        hier_sort.append(hierarchy[0][i])

    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i],True)
        approx = cv2.approxPolyDP(cnts_sort[i],0.01*peri,True)

        if ((size < 120000) and (size > 25000) and hier_sort[i][3] == -1) and (len(approx) == 4):
            cnts_is_card[i] = 1

    return cnts_sort, cnt_is_card

def preprocess_card(contour, image):
    qCard = Query_card()
    qCard.contour = contour

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01*peri,True)
    pts = np.float32(approx)
    qCard.corner_pts = pts
    
    x,y,w,h = cv2.boundingRect(contour)
    qCard.width, qCard.height = w,h

    average = np.sum(pts, axis=0)/len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    qCard.center = [cent_c, cent_y]

    qCard.warp = flattener(image, pts, w, h)

    Qcroner = qCard.warp[0:84, 0:32]
    Qcorner_zoom = cv2.resize(Qcorner, (0,0), fx = 4, fy = 4)

    white_level = Qcroner_zoom[15,int((32*4)/2)]
    thresh_level = white_level - 30
    if(thresh_level <=0):
        thresh_level = 1
    retval, query_thresh = cv2.threshold(Qcorner_zoom, thresh_level, 255, cv2.THRESH_BINARY_INV)

    Qrank = query_thresh[20:185, 0:128]
    Qcolour = query_thresh[186:336, 0:128]

    dummy, Qrank_cnts, hier = cv2.findContours(Qrank, cv2.RETR_REE, cv2.CHAIN_APPROX_SIMPLE)
    Qrank_cnts = sorted(Qrank_cnts, key = cv2.contourArea, reverse = True)

    if len(Qrank_cnts) != 0:
        x1,y1,w1,h1 = cv2.boundingRect(Qrank_cnts[0])
        Qrank_roi = Qrank[y1:y1+h1, x1:x1+w1]
        Qrank_sized = cv2.resize(Qrank_roi,(70,125),0,0)
        qCard.rank_img = Qrank_sized

    dummy, Qcolour_cnts, hier = cv2.findContours(Qcolour, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Qcolour_cnts = sorted(Qsuit_cnts, key = cv2.contourArea, reverse = True)

    if len(Qcolour_cnts) != 0:
        x2,y2,w2,h2 = cv2.boundingRect(Qcolour_cnts[0])
        Qcolour_roi = Qcolour[y2:y2+h2, x2:x2+w2]
        Qcolour_sized = cv2.resize(Qsuit_roi, (70,100),0,0)
        qCard.colour_img = Qcolour_sized
        
    return qCard

def match_card(qCard, Train, Train_colour):
    best_rank_match_diff = 10000
    best_colour_match_diff = 10000
    best_rank_match_name = 'Unknown'
    best_colour_match_name = 'Unknown'

    i = 0

    if (len(qCard.rank_img) != 0) and (len(qCard.colour.img) != 0):
        for Trank in Train:
            diff_img = cv2.abdsdiff(qCard.rank_img, Trank.img)
            rank_diff = int(np.sum(diff_img)/255)
            
            if rank_diff < best_rank_match_diff:
                best_rank_diff_img =  diff_img
                best_rank_match_diff = rank_diff
                best_rank_name = Trank.name

        for Tcolour in Train_colour:
            diff_img = cv2.abdsdiff(qCard.colour_img, Tcolour.img)
            colour_diff = int(np.sum(diff_img)/255)

            if colour_diff < best_colour_match_diff:
                best_colour_diff_img = diff_img
                best_colour_match_diff = colour_diff
                best_colour_name = Tcolour.name

        if (best_rank_match_diff < 2000):
            best_rank_match_name = best_rank_name

        if (best_colour_match_diff < 700):
            best_colour_match_name = best_colour_name

        return best_rank_match_name, best_colour_match_name, best_rank_match_diff, best_colour_match_diff
    
def draw_results(image, qCard):

    x = qCard.center[0]
    y = qCard.center[1]
    cv2.circle(image,(x,y),5,(255,0,0),-1)

    rank_name = qCard.best_rank_match
    colour_name = qCard.best_colour_name

    cv2.putText(image,(rank_name+' of'),(x-60, y-10), font,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,(rank_name+' of'),(x-60,y-10),font,1,(50,200,200),2,cv2.LINE_AA)

    cv2.putText(image,suit_name,(x-60,y+25),font,1,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,suit_name,(x-60,y+25),font,1,(50,200,200),2,cv2.LINE_AA)
    
    # Can draw difference value for troubleshooting purposes
    # (commented out during normal operation)
    #r_diff = str(qCard.rank_diff)
    #s_diff = str(qCard.suit_diff)
    #cv2.putText(image,r_diff,(x+20,y+30),font,0.5,(0,0,255),1,cv2.LINE_AA)
    #cv2.putText(image,s_diff,(x+20,y+50),font,0.5,(0,0,255),1,cv2.LINE_AA)

    return image
    
def falttener(image, pts, w, h):
    temp_rect = np.zeros((4,2), dtype = "float32")
    s = np.sum(pts, axis = 2)

    t1 = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = npdiff(pts, axis = -1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    if w <= 0.8*h:
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h: # If card is horizontally oriented
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    # If the card is 'diamond' oriented, a different algorithm
    # has to be used to identify which point is top left, top right
    # bottom left, and bottom right.
    
    if w > 0.8*h and w < 1.2*h: #If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0] # Top left
            temp_rect[1] = pts[0][0] # Top right
            temp_rect[2] = pts[3][0] # Bottom right
            temp_rect[3] = pts[2][0] # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0] # Top left
            temp_rect[1] = pts[3][0] # Top right
            temp_rect[2] = pts[2][0] # Bottom right
            temp_rect[3] = pts[1][0] # Bottom left
            
        
    maxWidth = 200
    maxHeight = 300

    # Create destination array, calculate perspective transform matrix,
    # and warp card image
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0, maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect,dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warp = cv2.cvtColor(warp,cv2.COLOR_BGR2GRAY)
    
    return warp
    
def colourdetect(img):

    crop(img)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Set range for red color and
    # define mask
    red_lower = np.array([0, 50, 20])
    red_upper = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
  
    # Set range for green color and 
    # define mask
    green_lower = np.array([45, 100, 50])
    green_upper = np.array([102, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
  
    # Set range for blue color and
    # define mask
    blue_lower = np.array([87, 150, 80])
    blue_upper = np.array([117, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # Set range for yellow color and
    # define mask
    yellow_lower = np.array([15, 90, 80])
    yellow_upper = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    #Adding all the colours to identify range of colours from red to yellow
    #in an image
    result = red_mask + green_mask + blue_mask + yellow_mask
    output = cv2.bitwise_and(img, img, mask = result)

    #Shows original image against the colour detection image
    cv2.imshow('Colour Detected', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

##for image in path:
##    
##    img = cv2.imread(image)
##    crop(img)

