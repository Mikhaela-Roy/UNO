import cv2
import numpy as np
import time
import with_fnctions
import os

img_width = 1280
img_height = 720
f_rate = 10

f_rate_calc = 1
freq = cv2.getTickFrequency()

font = cv2.FONT_HERSHEY_SIMPLEX

video = cv2.VideoCapture(0)
time.sleep(1)

path = os.path.dirname(os.path.abspath('C:/Users/Mikhaela Rain Roy/Desktop/UNO/Training'))
train_ranks = with_fnctions.load_rank(path + '/Training/')
train_colour = with_fnctions.load_colour(path + '/Training')

while (1):
    
    image = video.read()

    t1 = cv2.getTickCount()

    pre_proc = with_fnctions.preprocess(image)
    cnts_sort, cnt_is_card = with_fnctions.find_cards(pre_proc)

    if len(cnts_sort) != 0:
        cards = []
        k = 0

        for i in range(len(cnts_sort)):
            if (cnt_is_card[i] == 1):
                cards.append(with_fnctions.preprocess_card(cnts_sort[i], image))
                cards[k].best_rank_match, cards[k].best_colour_match, cards[k].rank_diff, cards[k].colour.diff = with_fnctions.match_card(cards[k], Train, Train_colour)
                image = with_fnctions.draw_results(image, cards[k])
                k += 1

        if (len(cards) != 0):
            temp_cnts = []
            for i in range(len(cards)):
                temp_cnts.append(cards[i].contour)
            cv2.drawContours(image, temp_cnts, -1,(255,0,0),2)

    cv2.imshow('Card detector', image)

    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    f_rate_calc = 1/time1

cv2.waitKey(0)
   
cv2.destroyAllWindows()
cv2.release()

        
