import numpy as np
from numpy import savetxt
from numpy import loadtxt
from sklearn import datasets
import matplotlib.pyplot as plt
import os
from PIL import Image

def ListofFiles(Dir):
    Files = []
    for root, dir_name, file_name in os.walk(Dir):
        for name in file_name:
            fullName = os.path.join(root, name)
            Files.append(fullName)
    return Files

FileList = ListofFiles(('C:/Users/Mikhaela Rain Roy/Desktop/UNO/UNO images'))

pixels = []

for file in FileList:
    Im= Image.open(file)
    pixels.append(list(Im.getdata()))
    
    print(pixels)

##pixels_arr = np.asarray(pixels)
##print(pixels_arr.shape)
##savetxt('uno_card.csv', pixels_arr, delimiter = ',')
##
##Image = loadtxt('uno_card.csv', delimiter=',')
##X = Image[0].reshape(28,28)
##print(Image.shape)
##plt.imshow(X)

