import cv2
import numpy as np
import mediapipe as mp
import time

class HandDetector():
    def __init__(self, mode=False, MaxHands=2, modelCmplx=1,
                    detectConfi=0.5, trackConfi=0.5):
        self.mode = mode
        self.MaxHands = MaxHands
        self.detectConfi = detectConfi
        self.trackConfi = trackConfi
        self.modelCmplx = modelCmplx

        self.mpH = mp.solutions.hands
        self.hands = self.mpH.Hands(self.mode, self.MaxHands, self.modelCmplx, 
                                    self.detectConfi, self.trackConfi)
        self.mpDraw = mp.solutions.drawing_utils

        self.fingerTips = [4, 8, 12, 16, 20]
    
    def findHand(self, frame, draw=True):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)
        if draw:
            if self.results.multi_hand_landmarks:
                for handLMS in self.results.multi_hand_landmarks: 
                    self.mpDraw.draw_landmarks(frame, handLMS, self.mpH.HAND_CONNECTIONS)
        return frame

    def findPos(self, frame, handNum=0, draw=True):
        #by default tracks thumb tip
        self.lmList = []
        if self.results.multi_hand_landmarks:
            selectedHand = self.results.multi_hand_landmarks[handNum]

            for id,lm in enumerate(selectedHand.landmark):
                height, width, channels = frame.shape
                cx, cy = int(lm.x*width), int(lm.y*height)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx,cy), 10, (255, 255, 0), cv2.FILLED)

        return self.lmList
    
    def cntFingersUp(self, imgFlip=False):
        fingers = []
        
        #for thumb finger
        if self.lmList[4][1] > self.lmList[3][1]:
            fingers.append(True)
        else:
            fingers.append(False)
        #for other remaining fingers
        for id in range(1,5):
            if self.lmList[self.fingerTips[id]][2] < self.lmList[self.fingerTips[id]-2][2]:
                fingers.append(True)
            else:
                fingers.append(False)
        if imgFlip:
            fingers[0] = not fingers[0]
        return fingers
            
def main():
    prevTime = curTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    
    while(True):
        ret, frame = cap.read()
        frame = detector.findHand(frame)
        pos = detector.findPos(frame)
        # if len(pos)!=0:
        #     print(pos[4]) #by default prints thumb tip position

        curTime = time.time()
        fps = 1/(curTime - prevTime)
        prevTime = curTime

        cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 
                        3, (255,0,255), 3)
        cv2.imshow('preview',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




if __name__ == '__main__':
    main()