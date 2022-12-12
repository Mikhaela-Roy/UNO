import os

path = './UNO_images/b0.jpg'


def card_type_name(path):
    
    card_type = path[-5:-4]
    if card_type == "R":
        print('r')
        card_type = 10;
    elif card_type == "S":
        card_type = 11;
    elif card_type == "T":
        card_type = 12;
    elif card_type == "B":
        card_type = 13;    
    elif card_type == "E":
        card_type = 14;  
    elif card_type == "F":
        card_type = 15;  
    elif card_type == "W":
        card_type = 16;  
    else:
        card_type = int(path[-5:-4])
    return card_type

card_type_name(path)

