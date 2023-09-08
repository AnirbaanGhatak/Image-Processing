# Name: Anirbaan Ghatak
# Roll no: C026
# Aim:To write a program to enhance the quality of an image by noise removal.
import numpy as np
import cv2 as cv
from google.colab.patches import cv2_imshow

img = cv.resize(cv.imread('asdfjk.jpg'), (256, 256))


def showImage(img):
    cv2_imshow(img)
    cv.waitKey(0)
    cv.destroyAllWindows()


grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
showImage(grayImg)

h2 = np.array([[1, 1], [1, -1]])
h = h2.copy()
for i in range(7):
    h = np.kron(h, h2)
hT = np.transpose(h)

hadamard = np.dot(np.dot(h, grayImg), hT).astype(np.uint8)

showImage(hadamard)

inverse = np.dot(np.dot(h, hadamard), hT)/(256**2)
showImage(inverse.astype(np.uint8))

rc = dict()
for i in range(len(h)):
    change = 0
    x = 1
    for j in h[i]:
        if x != j:
            x = j
            change += 1
    rc[i] = change

sort = sorted(rc.items(), key=lambda kv:
              kv[1])

walsh=h.copy()
for i in range(len(walsh)):
    index=sort[i][0]
    walsh[i]=h[index]
walsh=walsh.astype(np.uint8)

showImage(walsh)

walshT=np.dot(np.dot(walsh,grayImg),np.transpose(walsh)).astype(np.uint8)

showImage(walshT)

inverseWalsh=np.dot(np.dot(walsh,walshT),np.transpose(walsh))//(255**2)
showImage(inverseWalsh.astype(np.uint8))