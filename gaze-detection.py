#!/usr/bin/env python3
import cv2
import dlib
import numpy as np
import pyautogui
import time
import webbrowser

time_old = time.time()
def shape_to_np(shape, dtype="int"):
	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

all_left = list()
all_right = list()

def click_func(old_pos):
    new_pos = pyautogui.position()
    
    if new_pos==old_pos:
        pyautogui.click()
        time_old = time.time()



def contouring(thresh, mid, img, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        #print("qqq")
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
            flag=1
        else:
            flag=0
        old_pos=pyautogui.position()
        mouse_control(cx, cy, flag)
        if time.time()-time_old>8:
            click_func(old_pos)

       
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)

    except:
        pass
def mouse_control(cx, cy, flag):
    
    if flag==0:
        all_left.append([cx, cy])

    else:
        all_right.append([cx, cy])
        if len(all_right)>2:
            mid_new =[((all_right[len(all_right)-1])[0]+(all_left[len(all_left)-1])[0])/2,
                         ((all_right[len(all_right)-1])[1]+(all_left[len(all_left)-1])[1])/2]

            mid_old =[((all_right[len(all_right)-2])[0]+(all_left[len(all_left)-2])[0])/2,
                         ((all_right[len(all_right)-2])[1]+(all_left[len(all_left)-2])[1])/2]
            # mid_old =[((all_right[0])[0]+(all_left[0])[0])/2,
            #              ((all_right[0])[1]+(all_left[0])[1])/2]             
            #print(mid_new, mid_old)
            compare(mid_new, mid_old)

def compare(curr, prev):
    if curr[0]-prev[0]>5.0 and curr[1]-prev[1]<10.0:
        pyautogui.moveRel(40,0,duration=0.1)
    elif prev[0]-curr[0]>5.0 and prev[1]-curr[1]<10.0:
        pyautogui.moveRel(-40,0,duration=0.1)
    elif prev[1]-curr[1]>5.0:
        pyautogui.moveRel(0,-40,duration=0.1)
    elif curr[1]-prev[1]>5.0:
        pyautogui.moveRel(0,40,duration=0.1)



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_68.dat')

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0)
ret, img = cap.read()
thresh = img.copy()

#cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)


webbrowser.open('https://prami1806.github.io/Express/')

cv2.namedWindow('eyes', cv2.WINDOW_NORMAL)
cv2.resizeWindow('eyes', 400, 400)
cv2.moveWindow('eyes',0, 580)
while(True):
    
    ret, img = cap.read()
    count = 0
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    

    for rect in rects:
        if count%3 !=0:
            continue

        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = eye_on_mask(mask, left)
        mask = eye_on_mask(mask, right)
        mask = cv2.dilate(mask, kernel, 5)
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        #threshold = cv2.getTrackbarPos('threshold', 'image')
        threshold = 55
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2) #1
        thresh = cv2.dilate(thresh, None, iterations=4) #2
        thresh = cv2.medianBlur(thresh, 3) #3q
        thresh = cv2.bitwise_not(thresh)
        contouring(thresh[:, 0:mid], mid, img)
        contouring(thresh[:, mid:], mid, img, True)
        for (x, y) in shape[36:48]:
            cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
        count+=1
        
    
    cv2.imshow('eyes', img)
    
 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()
