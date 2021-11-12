# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
import pyfits
import glob
import numpy
from PIL import Image
from astropy.io import fits
import matplotlib.pyplot as plt

for index in range(1,226):
    index = index-1

    image = "/home/kupa/kupa/unetfiles/unet-Sydil/concentrated/actual/{}.png".format(index)
    image = cv2.imread(image)


    mask = "/home/kupa/kupa/unetfiles/unet-Sydil/concentrated/mask/{}model.png".format(index)
    mask = cv2.imread(mask)


    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred,60, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:

        (x, y, w, h) = cv2.boundingRect(c)
        print(x, y, w, h)
        cv2.rectangle(image, (x-5,y-5), (x+w+10,y+h+10), (0, 255, 0), 2)

    newimage = "{}blobs.png".format(index)
    cv2.imwrite(newimage, image)
