import numpy as np
from PIL import Image
from sporco import plot
import util
import matplotlib.pyplot as plt

def Image_In(num):
    if(num < 10):
        num = "00" + str(num)
    elif(10 <= num < 100):
        num = "0" + str(num)
    else:
        num = str(num)
    im = np.array(Image.open("data/foreman/foreman"+num+".png"))

    return util.normalize(im.flatten())



