import math

import mediapipe as mp
import cv2
import time

class HandDetection():
    def __init__(self, mode=False, max_hands=2, dconfidence=0.5, tconfidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.dconfidence = dconfidence
        self.tconfidence = tconfidence
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.dconfidence,
            min_tracking_confidence=self.tconfidence
        )
        self.tipid = [4, 8, 12, 16, 20]

    def findHands(self, img):
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgrgb)

        if self.result.multi_hand_landmarks:
            for hand_landmark in self.result.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    img, hand_landmark, self.mp_hands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, indx_hand=0):
        self.land_mark_list = []
        xlist=[]
        ylist=[]
        bbox=[]
        if self.result.multi_hand_landmarks:
            myhand = self.result.multi_hand_landmarks[indx_hand]
            for id, lm in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xlist.append(cx)
                ylist.append(cy)
                self.land_mark_list.append([id, cx, cy])
            xmin,xmax=min(xlist),max(xlist)
            ymin,ymax=min(ylist),max(ylist)
            bbox=xmin,ymin,xmax,ymax
        return self.land_mark_list,bbox

    def fingersUp(self):
        fingers = []
        if len(self.land_mark_list) == 0:
            return fingers

        # Thumb
        if self.land_mark_list[self.tipid[0]][1] > self.land_mark_list[self.tipid[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):
            if self.land_mark_list[self.tipid[id]][2] < self.land_mark_list[self.tipid[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def distance_btw_fingers(self,p1,p2,img):
        x1,y1=self.land_mark_list[p1][1],self.land_mark_list[p1][2]
        x2,y2=self.land_mark_list[p2][1],self.land_mark_list[p2][2]
        cx,cy=(x1+x2)//2,(y1+y2)//2
        length = math.hypot(x2 - x1, y2 - y1)
        return length,[x1,y1,x2,y2,cx,cy]
def main():
    cap = None
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            break
        cap.release()

    if not cap or not cap.isOpened():
        print("Error: Could not open any camera.")
        return

    ptime = 0
    detector = HandDetection()
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        img = detector.findHands(img)
        findpos,bbox = detector.findPosition(img)
        fingers = detector.fingersUp()
        if len(fingers)!=0:
            print(fingers)
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
