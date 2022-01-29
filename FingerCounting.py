from chardet import detect
import cv2
from cv2 import waitKey
import mediapipe as mp
import time
import HandTrackModule as htm

wCam, hCam = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.HandDetector(detectConfi=0.75)
fingerTips = [4, 8, 12, 16, 20] #landmark ids for all the finger tips

while True:
    succ, frame = cap.read()

    frame = detector.findHand(frame)
    lmList = detector.findPos(frame, draw=False)
    #print(lmList)
    
    if len(lmList)!=0:
        fingers = []
        #for thumb finger
        if lmList[4][1] > lmList[3][1]:
            fingers.append('Open')
        else:
            fingers.append('Closed')
        #for other remaining fingers
        for id in range(1,5):
            if lmList[fingerTips[id]][2] < lmList[fingerTips[id]-2][2]:
                fingers.append('Open')
            else:
                fingers.append('Closed')
        #print(fingers)
        totalFingersOpen = fingers.count('Open')
        #print(totalFingersOpen)
        cv2.rectangle(frame, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, str(totalFingersOpen), (43,375), cv2.FONT_HERSHEY_COMPLEX,
                        5, (255,0,0), 15)
    cv2.imshow('Preview', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break