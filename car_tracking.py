# import the necessary packages
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
import spidev
import RPi.GPIO as GPIO

def Average(lst):
    return int(sum(lst) / len(lst))
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--lab",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
ap.add_argument("-p", "--pi", help="raspi")
args = vars(ap.parse_args())
# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the

lower_lab = {'red':(165, 32, 2),  'blue':(100, 200, 45), 'yellow':(21, 120, 200), 'orange':(0, 155, 55), 'teal': (69, 56, 187)}
upper_lab = {'red':(180,255,255), 'blue':(117,255,255), 'yellow':(54,255,255),'orange':(10,255,255), 'teal': (107, 160, 255)}
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

#Setup SPI comm
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 390625
# Split an integer input into a two byte array to send via SPI
def write_pot(input):
    #   msb = input >> 8
    #    lsb = input & 0xFF
    spi.xfer([input])


GPIO.setmode(GPIO.BCM)
#yellow
GPIO.setup(24, GPIO.OUT)
GPIO.setup(25, GPIO.OUT)
#blue
GPIO.setup(26, GPIO.OUT)
GPIO.setup(8, GPIO.OUT)
GPIO.output(24, 0)
GPIO.output(25, 0)
GPIO.output(26, 0)
GPIO.output(8, 0)

def write_gpio(car, angle, direction):
    L = 0
    R = 0
    if car == "yellow":
        L = 24
        R = 25
    else:
        L = 26
        R = 8

    if angle < 30:
        GPIO.output(L, 0)
        GPIO.output(R, 0)
        print("sending 0")
    elif direction == "L":
        GPIO.output(L, 1)
        GPIO.output(R, 0)
        print("sending 1")

    elif direction == "R":
        GPIO.output(L, 0)
        GPIO.output(R, 1)
        print("sending 2")

    else:
        GPIO.output(L, 0)
        GPIO.output(R, 0)
        print("sending 0")

# keep looping
while True:
    #Grab a frame of the video feed
    frame = vs.read()

    if frame is None:
        break

    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=400)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    rect1_center = None #The cennter of car1
    rect2_center = None #The cennter of car1
    ball_center = None #The center of the game ball
    rect1_angle = None #Not sure
    rect1_to_ball = None #The angle between the car1 and the ball
    yellow_front_car = None #The center of the cricle at the front fo the car
    blue_front_car = None #The center of the cricle at the front fo the car

    upper_hsv = upper
    if args.get("lab", True):
        upper_hsv  = upper_lab


    for color, val in upper_hsv.items():
        lower_hsv = lower
        if args.get("lower", True):
            lower_hsv = lower_lab
        mask = cv2.inRange(hsv, lower_hsv[color], upper_hsv[color])
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # only proceed if at least one contour was found
        if len(cnts) > 0:

            #Find the contours of the car
            if color == "red":
                minimum_size = 300
                areaArray = []
                for i, c in enumerate(cnts):
                    area = cv2.contourArea(c)
                    areaArray.append(area)
                sorteddata = sorted(zip(areaArray, cnts), key=lambda x: x[0], reverse=True)

                top_cnts = sorteddata[0][1]
                second = []
                if len(cnts) > 1:
                    second = sorteddata[1][1]

                if cv2.contourArea(top_cnts) > minimum_size :
                    rect = cv2.minAreaRect(top_cnts)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)

                    cv2.drawContours(frame,[box],0,(0,255,0),2)
                    rect1_angle = rect[2]
                    rect1_center_float = rect[0]
                    rect1_center = (int(rect1_center_float[0]), int(rect1_center_float[1]))


                if len(cnts) > 1:
                    if cv2.contourArea(second) > minimum_size:
                        rect2 = cv2.minAreaRect(second)
                        rect2_center_float = rect[0] 
                        rect2_center = (int(rect2_center_float[0]), int(rect2_center_float[1]))
                        box2 = cv2.boxPoints(rect2)
                        box2 = np.int0(box2)
                        cv2.drawContours(frame,[box2],0,(0,255,0),2)

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
                yellow_front_car = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                cv2.circle(frame, (int(x), int(y)), int(radius), colors[color], 2)
            elif color == "teal":
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                blue_front_car = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                cv2.circle(frame, (int(x), int(y)), int(radius), colors[color], 2)
    if not ball_center:
        write_gpio("blue", 0, "L")
        write_gpio("yellow", 0, "L")
    else:
        if rect1_center and yellow_front_car:
            #check if the smaller rectangle is closer
            yellow_center = rect1_center
            if rect2_center:
                length1 = np.linalg.norm(np.array(rect1_center) - np.array(yellow_front_car))
                length2 = np.linalg.norm(np.array(rect2_center) - np.array(yellow_front_car))
                if length2 < length1:
                    yellow_center = rect2_center


            cv2.line(frame, yellow_center, ball_center, (0, 255, 0), 3)
            cv2.line(frame, yellow_center, yellow_front_car, (0, 255, 0), 3)
            b = yellow_center
            b += (0,)
            c = ball_center
            c += (0,)
            a = yellow_front_car
            a +=(0,)
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)

            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)
            direction = np.cross(ba, bc)[2]

            dir = 'L' if direction > 0 else 'R'
            angle = np.degrees(angle)
            print(angle)
            print(dir)
            write_gpio("yellow", angle, dir)
            cv2.putText(frame,dir + ' ' + str(angle),
                (Average([yellow_center[0], ball_center[0]]), Average([yellow_center[1],ball_center[1]])),
                font,
                fontScale,
                fontColor,
                lineType)
        if rect1_center and blue_front_car:
            blue_center = rect1_center
            if rect2_center:
                print(blue_front_car)
                print(rect1_center) 
                print(rect1_center) 
                length1 = np.linalg.norm(np.array(rect1_center) - np.array(blue_front_car))
                length2 = np.linalg.norm(np.array(rect2_center)- np.array(blue_front_car))
                if length2 < length1:
                    blue_center = rect2_center


            cv2.line(frame, blue_center, ball_center, (0, 255, 0), 3)
            cv2.line(frame, blue_center, blue_front_car, (0, 255, 0), 3)
            b = blue_center
            b += (0,)
            c = ball_center
            c += (0,)
            a = blue_front_car
            a +=(0,)
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)

            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)
            direction = np.cross(ba, bc)[2]

            dir = 'L' if direction > 0 else 'R'
            angle = np.degrees(angle)
            print(angle)
            print(dir)
            write_gpio("blue", angle, dir)
            cv2.putText(frame,dir + ' ' + str(angle),
                (Average([blue_center[0], ball_center[0]]), Average([blue_center[1],ball_center[1]])),
                font,
                fontScale,
                fontColor,
                lineType)
        if not blue_front_car:
                write_gpio("blue", 0, "L")
        if not yellow_front_car:
                write_gpio("yellow", 0, "L")

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
