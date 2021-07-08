#conda create --name tesseract
#cd datafiles/tesseract
#conda activate tesseract
# C:\Program Files\Tesseract-OCR


# module
import cv2 
import numpy as np 
import pytesseract 
import os 
import re

# link tesseract with pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


# region of intrest getting with help of ROI.py
ROI = [[(306, 104), (468, 122), 'text', 'customer_name'], 
[(4, 276), (120, 294), 'text', 'product name'], 
[(980, 462), (1038, 490), 'text', 'price']]


## read image and change in grayscale
img1 = cv2.imread('sample_reciept.png')
imgQ = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)


# detector
# this is here just to get insight we use them in case we have many images 
# then we use brutforce with them to detect headings 
# also use birdsview if image is taken with diffrent angle
orb = cv2.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(imgQ , None)
impkp1 = cv2.drawKeypoints(imgQ, kp1, None)


#cv2.imshow('output1', impkp1)
#cv2.imshow('output',imgQ)
#cv2.waitKey(0)

## cloning the image
imgclone = imgQ.copy()
imgmask = np.zeros_like(imgclone)


## draw rectangle on ROI and crop that area
myData = []
for x,r in enumerate(ROI):
    cv2.rectangle(imgmask, (r[0][0],r[0][1]), (r[1][0],r[1][1]), (0,255,0), cv2.FILLED)
    imgclone = cv2.addWeighted(imgclone, 0.99, imgmask, 0.1, 0)

    imgcrop = imgQ[r[0][1]:r[1][1], r[0][0]:r[1][0]]
    cv2.imshow(str(x), imgcrop)

    ## feed cropped image to pytesseract
    if r[2] == 'text':
        print(f'{r[3]} :{pytesseract.image_to_string(imgcrop)}')
        myData.append(pytesseract.image_to_string(imgcrop))
        #print(myData)

## save file in excel / csv
with open('output.csv', 'a+')as f:
    for a, data in enumerate(myData):
        data = re.sub('\xad(\x0c)*','',data)
        data = re.sub(r'\W+','',data) 
        f.write((str(data)+ ','))
    f.write('\n')


cv2.imshow('output',imgclone)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()