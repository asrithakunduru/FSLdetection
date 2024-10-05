import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

fslcap = cv2.VideoCapture(0)
fsldetector = HandDetector(maxHands=1)
fsloffset = 20
fslimgSize = 224

fslfolder = "data1/train/T"
fslcounter = 0
while True:
    success, fslimg = fslcap.read()
    hands, fslimg = fsldetector.findHands(fslimg, flipType=False)
    if hands:
        hand = hands[0]
        fslx, fsly, fslw, fslh = hand['bbox']

        fslimgwhite = np.ones((fslimgSize, fslimgSize, 3), np.uint8)*255
        fslimgCrop = fslimg[fsly-fsloffset:fsly + fslh+fsloffset, fslx-fsloffset:fslx + fslw+fsloffset]

        fslimgCropShape = fslimgCrop.shape

        fslaspectRatio = fslh/fslw

        if  fslaspectRatio >1:
            fslk = fslimgSize/fslh
            fslwCal = math.ceil(fslk*fslw)
            fslimgResize = cv2.resize(fslimgCrop, (fslwCal, fslimgSize))
            fslimgResizeShape = fslimgResize.shape
            fslwGap = math.ceil((fslimgSize-fslwCal)/2)
            fslimgwhite[:, fslwGap:fslwCal+fslwGap] = fslimgResize

        else:
            fslk = fslimgSize / fslw
            fslhCal = math.ceil(fslk * fslh)
            fslimgResize = cv2.resize(fslimgCrop, (fslimgSize, fslhCal))
            fslimgResizeShape = fslimgResize.shape
            fslhGap = math.ceil((fslimgSize - fslhCal) / 2)
            fslimgwhite[fslhGap:fslhCal + fslhGap, :] = fslimgResize

        cv2.imshow("ImageCrop", fslimgCrop)
        cv2.imshow("Imagewhite", fslimgwhite)

    cv2.imshow("Image", fslimg)
    fslkey = cv2.waitKey(1)
    if fslkey == ord("s"):
        fslcounter += 1
        cv2.imwrite(f'{fslfolder}/Image_{time.time()}.jpg', fslimgwhite)
        print(fslcounter)