import numpy as np
import cv2
import HandDetection as hd
import autopy
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

plocX, plocY = 0, 0
clocX, clocY = 0, 0

wCam, hCam = 640, 480
wscr, hscr = autopy.screen.size()
print(wscr, hscr)
detector = hd.HandDetection(max_hands=1)

cap = cv2.VideoCapture(0)

# Volume control setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

# Time delay variables
click_delay = 1  # 1 second delay
last_click_time = time.time()

while cap.isOpened():
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist, bbox = detector.findPosition(img)

    if len(lmlist) != 0:
        x1, y1 = lmlist[4][1:]
        x2, y2 = lmlist[8][1:]

    # Checking for fingers up
    fingers = detector.fingersUp()
    cv2.rectangle(img, (100, 100), (wCam - 100, hCam - 50), (0, 255, 0), 2)

    if len(fingers) != 0:
        if fingers[0] == 0 and fingers[1] == 1:
            # Convert Coordinates
            x3 = np.interp(x2, (100, wCam - 100), (0, wscr))
            y3 = np.interp(y2, (100, hCam - 100), (0, hscr))

            clocX = plocX + (x3 - plocX) / 5
            clocY = plocY + (y3 - plocY) / 5

            autopy.mouse.move(wscr - clocX, clocY)
            plocX, plocY = clocX, clocY

    if len(fingers) != 0:
        if fingers[0] == 1 and fingers[1] == 1:
            length, info = detector.distance_btw_fingers(4, 8, img)
            # Check if enough time has passed since the last click
            if length < 25 and (time.time() - last_click_time) > click_delay:
                autopy.mouse.click()
                last_click_time = time.time()

    # Volume control based on distance between thumb and index finger
    # if len(fingers)!=0:
    #     if fingers[0] == 1 and fingers[2] == 1:
    #         length, info = detector.distance_btw_fingers(4, 12, img)
    #         if length > 20:
    #             vol = np.interp(length, [25, 150], [minVol, maxVol])
    #             volume.SetMasterVolumeLevel(vol, None)
    #
    #             # Visual feedback for volume control
    #             cv2.putText(img, f'Volume: {int(vol)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Image", img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
