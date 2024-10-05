import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from tensorflow.keras.models import load_model
import tensorflow as tf
import keras
fslcap = cv2.VideoCapture(0)
fsldetector = HandDetector(maxHands=1)
fslmodel = load_model("Model1/resnet50_saved_model")
fsloffset = 20
fslimgSize = 224
fsl_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
              10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
              19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
while True:
    success, fslimg = fslcap.read()
    fslimgOutput = fslimg.copy()
    hands, fslimg = fsldetector.findHands(fslimg)
    if hands:
        hand = hands[0]
        fslx, fsly, fslw, fslh = hand['bbox']

        fslimgwhite = np.ones((fslimgSize, fslimgSize, 3), np.uint8) * 255
        fslimgCrop = fslimg[fsly - fsloffset:fsly + fslh + fsloffset, fslx - fsloffset:fslx + fslw + fsloffset]

        fslimgCropShape = fslimgCrop.shape

        fslaspectRatio = fslh / fslw

        if fslaspectRatio > 1:
            fslk = fslimgSize / fslh
            fslwCal = math.ceil(fslk * fslw)
            fslimgResize = cv2.resize(fslimgCrop, (fslwCal, fslimgSize))
            fslimgResizeShape = fslimgResize.shape
            fslwGap = math.ceil((fslimgSize - fslwCal) / 2)
            fslimgwhite[:, fslwGap:fslwCal + fslwGap] = fslimgResize
            fslpredictions = fslmodel.predict(np.expand_dims(fslimgwhite, axis=0))  # Pass imgwhite for fslprediction
            fslpredicted_label = np.argmax(fslpredictions)
            fsl_sign = fsl_labels[fslpredicted_label]

        else:
            fslk = fslimgSize / fslw
            fslhCal = math.ceil(fslk * fslh)
            fslimgResize = cv2.resize(fslimgCrop, (fslimgSize, fslhCal))
            fslimgResizeShape = fslimgResize.shape
            fslhGap = math.ceil((fslimgSize - fslhCal) / 2)
            fslimgwhite[fslhGap:fslhCal + fslhGap, :] = fslimgResize
            fslpredictions = fslmodel.predict(np.expand_dims(fslimgwhite, axis=0))  # Pass imgwhite for fslprediction
            fslpredicted_label = np.argmax(fslpredictions)
            fsl_sign = fsl_labels[fslpredicted_label]

        cv2.putText(fslimgOutput, f'FSL: {fsl_sign}', (fslx, fsly - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
        cv2.rectangle(fslimgOutput, (fslx - fsloffset, fsly - fsloffset), (fslx + fslw + fsloffset, fsly + fslh + fsloffset), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", fslimgCrop)
        cv2.imshow("Imagewhite", fslimgwhite)

    cv2.imshow("Image", fslimgOutput)
    fslkey = cv2.waitKey(1)