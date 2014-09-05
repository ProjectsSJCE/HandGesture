from PIL import Image
from numpy import *
from pylab import *
import os
import cv
import SimpleCV
import time
number = 1
directory_name = "complexb"
filenames = os.listdir(directory_name)
fig = figure()
for filename in filenames:
    print filename
    image = str(directory_name) + "/" + str(filename) 
    img = Image.open(image)
#    border = (70, 40, 420, 450)
#    border = (40, 10, 500, 550)
#    img = img.crop(border)
#    img = img.resize((100,100), Image.ANTIALIAS)
    img.save("complexB/B" + str(number) + ".jpg")
    number += 1
#image = "data1/img182.jpg"
#img = Image.open(image)
#border = (70, 50, 420, 450)
#img = img.crop(border)
#img = img.resize((100,100), Image.ANTIALIAS)
#imshow(array(img))
#fig.show()
#a = raw_input("as")
#close()

