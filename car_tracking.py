import io
from collections import deque
from imutils.video import VideoStream
from picamera.array import PiRGBArray
import picamera
from picamera import PiCamera
import numpy as np
import argparse
import cv2
import imutils
import time
import math
from imutils.video.fps import FPS
from imutils.video.pivideostream import PiVideoStream
import RPi.GPIO as GPIO

def Average(lst):
    return int(sum(lst) / len(lst))

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
ap.add_argument("-p", "--pi", help="raspi")
args = vars(ap.parse_args())
# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the

lower = {'red':(165, 32, 2),  'blue':(100, 200, 45), 'yellow':(20, 66, 160), 'orange':(0, 119, 50), 'teal': (81, 66, 204)}
upper = {'red':(180,255,255), 'blue':(117,255,255), 'yellow':(54,255,255),'orange':(10,255,255), 'teal': (100, 160, 255)}
colors = {'red':(0,0,255), 'green':(0,255,0), 'teal':(255,0,0), 'yellow':(0, 255, 217), 'orange':(0,140,255)}

# Write some Text
font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

# allow the camera or video file to warm up
vs =PiVideoStream().start()
time.sleep(2.0)

#Setup GPI output
GPIO.setmode(GPIO.BCM)
GPIO.setup(24, GPIO.OUT)
GPIO.setup(25, GPIO.OUT)
GPIO.output(24, 0)
GPIO.output(25, 0)

def write_GPIO(car, dir, angle):
    L = 24
    R = 25
    if car == 2:
        L = 26
        R = 8

    if( angle < 30 ):
        GPIO.output(L, 0)
        GPIO.output(R, 0)
        print("sending: 0")
    elif dir == 'L':
        #write_pot(0x31)
        GPIO.output(L, 1)
        GPIO.output(R, 0)
        print("sending: 1")
    elif dir == 'R':
        GPIO.output(L, 0)
        GPIO.output(R, 1)
        #write_pot(0x32)
        print("sneding: 2")
    else:
        #write_pot(0x00)
        print("sending: 0 cuz none")

# keep looping
while True:
    #Grab a frame of the video feed
    frame = vs.read()
    if frame is None:
        break

    # resize the frame, blur it, and convert it to the HSV color thing
    frame = imutils.resize(frame, width=400)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    rect1_center = None #The cennter of car1
    rect2_center = None
    ball_center = None #The center of the game ball
    rect1_to_ball = None #The angle between the car1 and the ball
    front_car_1 = (1000000,1000000) #The center of the cricle at the front fo the car
    front_car_2 = (1000000,1000000) #The center of the circle at the front of the second car

    dist1 = None #distance from closest car to front1
    dist2 = None #distance from closest car to front1


    for color, val in upper.items():

        mask = cv2.inRange(hsv, lower[color], upper[color])
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            #Find the contours of the cars
            if color == "red":
                minimum_size = 300
                areaArray = []
                for i, c in enumerate(cnts):
                    area = cv2.contourArea(c)
                    areaArray.append(area)
                sorteddata = sorted(zip(areaArray, cnts), key=lambda x: x[0], reverse=True)

                top_cnts = sorteddata[0][1]
                second = []
                if cv2.contourArea(top_cnts) > minimum_size :
                    rect = cv2.minAreaRect(top_cnts)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)

                    cv2.drawContours(frame,[box],0,(0,255,0),2)
                    rect1_center_float = rect[0]
                    rect1_center = (int(rect1_center_float[0]), int(rect1_center_float[1]))


                if len(cnts) > 1:
                    second = sorteddata[1][1]
                    if cv2.contourArea(second) > minimum_size:
                        rect2 = cv2.minAreaRect(second)
                        box2 = cv2.boxPoints(rect2)
                        box2 = np.int0(box2)
                        cv2.drawContours(frame,[box2],0,(0,255,0),2)
                        rect2_center_float = rect[0]
                        rect2_center = (int(rect2_center_float[0]), int(rect2_center_float[1]))
            #Find ball
            elif color == "orange":
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                ball_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                cv2.circle(frame, (int(x), int(y)), int(radius), (0,0,255), 2)
            elif color == "yellow":
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                front_car_1 = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                cv2.circle(frame, (int(x), int(y)), int(radius), colors[color], 2)
            elif color == "teal":
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                front_car_2 = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                M = cv2.moments(c)
                cv2.circle(frame, (int(x), int(y)), int(radius), colors[color], 2)

    print(rect1_center)
    if ball_center:
            #do nothing
        if rect1_center and front_car_1 and rect2_center and front_car_2:
            #find closest car to ball1
            a = front_car_1
            a +=(0,)
            b = rect1_center
            b += (0,)
            c = ball_center
            c += (0,)
            d = front_car_2
            d +=(0,)
            e = rect2_center
            e += (0,)

            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            d = np.array(d)
            e = np.array(e)

            dist1 = np.linalg.norm(a - b)
            temp = np.linalg.norm(a - e)
            #vector that corresponds to car 1
            vec_1 = b
            vec_2 = e

            if temp < dist1:
                vec_1 = e
                vec_2 = b
            #car 1 vector
            car1 = a - vec1
            #ball to car 1 vector
            car1_to_ball = c - vec1
            #car 2 vector
            car2 = d - vec2
            #ball to car 2 vector
            car2_to_ba = c - vec2
            cosine_angle_1 = np.dot(car1, car1_to_ball) / (np.linalg.norm(car1) * np.linalg.norm(car1_to_ball))
            cosine_angle_2 = np.dot(car2, car2_to_ball) / (np.linalg.norm(car2) * np.linalg.norm(car2_to_ball))
            angle1 = np.arccos(cosine_angle_1)
            direction1 = np.cross(car1, car1_to_ball)[2]
            angle2 = np.arccos(cosine_angle_2)
            direction2 = np.cross(car2, car2_to_ball)[2]

            dir1 = 'L' if direction1 > 0 else 'R'
            dir2 = 'L' if direction2> 0 else 'R'
            write_GPIO(1, dir1, angle1)
            write_GPIO(2, dir2 ,angle2)

        #if car 1 only
    elif rect1_center and ( front_car_1 or front_car_2 ):
            #find closest car to ball1
            a = front_car_1
            a +=(0,)
            b = rect1_center
            b += (0,)
            c = ball_center
            c += (0,)
            d = front_car_2
            d +=(0,)


            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            c = np.array(d)

            #Check which front of the car to pick
            dist1 = np.linalg.norm(a - b)
            temp = np.linalg.norm(d - b)
            #vector that corresponds to car 1
            vec_1 = b
            vec_2 = b
            front = a

            #the valid car front, closests to the only rect
            car_num = 1
            if temp < dist1:
                front = d
                car_num = 2

            car = front - vec_1
            #ball to car 1 vector
            car1_to_ball = c - vec1

            #This can be car1 or car2
            cosine_angle_1 = np.dot(car, car1_to_ball) / (np.linalg.norm(car1) * np.linalg.norm(car1_to_ball))
            angle1 = np.arccos(cosine_angle_1)
            direction1 = np.cross(car1, car1_to_ball)[2]

            dir1 = 'L' if direction1 > 0 else 'R'
            if car_num == 1:
                write_GPIO(1, dir1, angle1)
                write_GPIO(2, dir1, 0)
            if car_num == 2:
                write_GPIO(1, dir1 ,0)
                write_GPIO(2, dir1 ,angle1)
    elif rect2_center and ( front_car_1 or front_car_2 ):
            #find closest car to ball1
            a = front_car_1
            a +=(0,)
            b = rect2_center
            b += (0,)
            c = ball_center
            c += (0,)
            d = front_car_2
            d +=(0,)


            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            c = np.array(d)

            #Check which front of the car to pick
            dist1 = np.linalg.norm(a - b)
            temp = np.linalg.norm(d - b)
            #vector that corresponds to car 1
            vec_1 = b
            vec_2 = b
            front = a

            #the valid car front, closests to the only rect
            car_num = 1
            if temp < dist1:
                front = d
                car_num = 2

            car = front - vec_1
            #ball to car 1 vector
            car1_to_ball = c - vec1

            #This can be car1 or car2
            cosine_angle_1 = np.dot(car, car1_to_ball) / (np.linalg.norm(car1) * np.linalg.norm(car1_to_ball))
            angle1 = np.arccos(cosine_angle_1)
            direction1 = np.cross(car1, car1_to_ball)[2]

            dir1 = 'L' if direction1 > 0 else 'R'
            if car_num == 1:
                write_GPIO(1, dir1, angle1)
                write_GPIO(2, dir1, 0)
            if car_num == 2:
                write_GPIO(1, dir1 ,0)
                write_GPIO(2, dir1 ,angle1)

    else:
        #write zeroes
        write_GPIO(0, 'L', 0)
        write_GPIO(1, 'L', 0)


        cv2.putText(frame,dir + ' ' + str(angle),
            (Average([rect1_center[0], ball_center[0]]), Average([rect1_center[1],ball_center[1]])),
            font,
            fontScale,
            fontColor,
            lineType)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
if not args.get("video", False):
    vs.stop()
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()
