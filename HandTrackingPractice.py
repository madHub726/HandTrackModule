import cv2 #Computer Vision Module
import mediapipe as mp #INbuilt and readymade hand recognition
import time #In order to show frame rate in real time if required

# Open the device at the ID 0

cap = cv2.VideoCapture(0)
mpH = mp.solutions.hands #Initializing the hand recognition patterns
hands = mpH.Hands() #hands object
mpDraw = mp.solutions.drawing_utils #in order to draw the landmarks
#Check whether user selected camera is opened successfully.
if not (cap.isOpened()):
    print('Could not open video device')
prevTime = curTime = 0

while(True):
    ret, frame = cap.read() # Capture frame-by-frame
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert to RGB format
    results = hands.process(imgRGB) #process the image

    if results.multi_hand_landmarks: #check if hands are detected or not in the video feed
        for handLMS in results.multi_hand_landmarks: #iterate  for each hand detected
            for id,lm in enumerate(handLMS.landmark): #capture each hand and its id & landmarks data
                #print(id,lm) #these values are in decimal places...
                height, width, channels = frame.shape #get the captured image dimensions
                cx, cy = int(lm.x*width), int(lm.y*height) #this will give us the central pixel locations
                print(id, cx, cy)
                
                #Now in order to highlight a certain landmark... we will do the following
                if id==4: #choose the id number to be highlighted... this is id of tip of thumb
                    cv2.circle(frame, (cx,cy), 25, (255, 255, 0), cv2.FILLED) #draw a big circle

            mpDraw.draw_landmarks(frame, handLMS, mpH.HAND_CONNECTIONS) #draw the landmark points on the image
    
    curTime = time.time()
    fps = 1/(curTime - prevTime)
    prevTime = curTime

    cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 
                    3, (255,0,255), 3)
    cv2.imshow('preview',frame)  # Display the resulting frame

    if cv2.waitKey(1) & 0xFF == ord('q'):  #Waits for a user input to quit the application
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()