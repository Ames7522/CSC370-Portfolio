# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 20:18:16 2022

@author: nhowe
"""

# from https://www.pluralsight.com/guides/importing-image-data-into-numpy-arrays
# check Pillow version number
import PIL
print('Pillow Version:', PIL.__version__)

# load and show an image with Pillow
from PIL import Image
# Open the image form working directory
image = Image.open('koala.jfif')
# summarize some details about the image
print(image.format)
print(image.size)
print(image.mode)
# show the image
image.show()

# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot as plt
# load image as pixel array
image = image.imread('koala.jfif')
# summarize shape of the pixel array
print(image.dtype)
print(image.shape)
# display the array of pixels as an image
plt.imshow(image)
plt.show()

# converting to NumPy array
from PIL import Image
from numpy import asarray
# load the image
image = Image.open('koala.jfif')
# convert image to numpy array
data = asarray(image)
print(type(data))
# summarize shape
print(data.shape)

# display
plt.imshow(data)
plt.axis('off')  # don't show axis labels
plt.show()

# create Pillow image
image2 = Image.fromarray(data)
print(type(image2))

# summarize image details
print(image2.mode)
print(image2.size)

# show red, green, blue components separately
# set default colormap
import matplotlib as mpl
mpl.rc('image', cmap='gray')
# mpl.rc('image', cmap='turbo')  # false color
# other possibilities:  inferno, turbo (like jet), magma, plasma 
# see https://matplotlib.org/stable/tutorials/colors/colormaps.html

def imshow(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
imshow(data[:,:,0])
imshow(data[:,:,1])
imshow(data[:,:,2])

# read directly to gray image and save again
import numpy as np
im = np.array(Image.open('koala.jfif').convert('L')) #you can pass multiple arguments in single line
# for modes see https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
print(type(im))
gr_im= Image.fromarray(im).save('gr_koala.png')

# show part of image

# using OpenCV
import cv2
im = cv2.imread('koala.jfif')
img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)   # BGR -> RGB
cv2.imwrite('opncv_koala.png', img) # colors reversed due to default BGR assumption
print (type(img))
img2 = im[:,:,[2,1,0]]  # another way
np.all(img2==img)  # true
np.all(img==data)  # true

# masking and components
fdata = img.astype(float)/255 # convert to float scaled 0 to 1
mask = fdata[:,:,2]-fdata[:,:,0]>0
imshow(mask)
simg = img.copy()
simg[~mask,:] = 0
imshow(simg)

def imfill(mask):
    (npix,ff,ff2,dim) = cv2.floodFill(np.uint8(mask),cv2.copyMakeBorder(np.uint8(mask),1,1,1,1,0,0),(0,0),1)
    return ~(ff-mask>0) # cv2.bitwise_not(ff-mask)
    
# cleaning up the blanket component
(numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(np.uint8(mask),connectivity=4,ltype=cv2.CV_32S)
imshow(labels==1)
mask2 = labels==1
simg = img.copy()
simg[~mask2,:] = 0
imshow(simg)
mask3 = imfill(mask2)
simg = img.copy()
simg[~mask3,:] = 0
imshow(simg)

# experiments with cois image
coins = cv2.imread('coins.png',cv2.IMREAD_GRAYSCALE)
imshow(coins)
print(coins.shape)
thr,cmask = cv2.threshold(coins,0,255,cv2.THRESH_OTSU)
imshow(cmask)
fcmask = imfill(cmask>thr)
imshow(fcmask)
imshow(coins*fcmask)

(numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(np.uint8(cmask),connectivity=4,ltype=cv2.CV_32S)
imshow(labels==1)
for i in range(1,numLabels):
    imshow(coins[stats[i,1]:stats[i,1]+stats[i,2],stats[i,0]:stats[i,0]+stats[i,2]])

# combining images
coins2 = cv2.copyMakeBorder(np.repeat(coins[:450,:,np.newaxis],3,axis=2),0,0,230,0,0)
cmask2 = cv2.copyMakeBorder(np.uint8(fcmask[:450,:]),0,0,230,0,0)>0
simg = img.copy()
simg[cmask2] = coins2[cmask2]
imshow(simg)