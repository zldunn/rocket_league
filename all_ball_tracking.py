# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import math

def Average(lst):
	return int(sum(lst) / len(lst))
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())
# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
redLower = (0, 50, 50)
redUpper = (10, 255, 255)

yellowLower = (20, 59, 100)
yellowUpper = (54, 255, 255)
orangeLower = (170, 100, 50)
orangeUpper = (180, 255, 255)
# define the lower and upper boundaries of the colors in the HSV color space
lower = {'red':(166, 84, 141), 'green':(66, 122, 129), 'blue':(97, 100, 117), 'yellow':(23, 59, 119)}#, 'orange':(5, 100, 140)}
upper = {'red':(186,255,255), 'green':(86,255,255), 'blue':(117,255,255), 'yellow':(54,255,255)}#,'orange':(15,255,255)}
colors = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0), 'yellow':(0, 255, 217), 'orange':(0,140,255)}

pts = deque(maxlen=args["buffer"])

curr_quad = 1
prev_quad = 1

prev_angle = 0
clock_wise = {
	1:2,
	2:3,
	3:4,
	4:1
}
c_clock_wise = {
	1:2,
	2:3,
	3:4,
	4:1
}

# Write some Text

font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

# keep looping
while True:
	# grab the current frame
	frame = vs.read()

	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break

	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	# construct a mask for the color yellow and red, then perform
	rect1_center = None
	ball_center = None
	rect1_angle = None
	rect1_to_ball = None
	for color, val in upper.items():

		mask = cv2.inRange(hsv, lower[color], upper[color])
		mask = cv2.erode(mask, None, iterations=2)
		mask = cv2.dilate(mask, None, iterations=2)

		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)

		cnts = imutils.grab_contours(cnts)
		# only proceed if at least one contour was found
		if len(cnts) > 0:

			if color == "red":
				minimum_size = 3000
				print(cnts)
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
					rect1_center = (Average([x[0] for x in box]), Average([y[1] for y in box]))
					rect1_angle = rect[2]
					if(rect[1][0] < rect[1][1]):
						rect1_angle += 90
					rect1_angle_opp = rect1_angle
					if rect1_angle > 0:
						rect1_angle_opp -= 180
					if rect1_angle < 0:
						rect1_angle_opp  +=180
					print("opp: " + str(rect1_angle_opp))
					rect1_angle = min([rect1_angle, rect1_angle_opp], key=lambda x:abs(x-prev_angle))
					print(rect1_angle)

				if len(cnts) > 1:
					if cv2.contourArea(second) > minimum_size:
						rect2 = cv2.minAreaRect(second)
						box2 = cv2.boxPoints(rect2)
						box2 = np.int0(box2)
						cv2.drawContours(frame,[box2],0,(0,255,0),2)
			else:
				c = max(cnts, key=cv2.contourArea)
				((x, y), radius) = cv2.minEnclosingCircle(c)
				M = cv2.moments(c)
				ball_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
				cv2.circle(frame, (int(x), int(y)), int(radius), colors[color], 2)

	if rect1_center and ball_center:
		cv2.line(frame, rect1_center, ball_center, (0, 255, 0), 3)
		rect1_to_ball = math.tan(rect1_center[1] - ball_center[1]/ rect1_center[0] - ball_center[0]) - rect1_angle
		cv2.putText(frame,str(rect1_to_ball),
		    (Average([rect1_center[0], ball_center[0]]), Average([rect1_center[1],ball_center[1]])),
		    font,
		    fontScale,
		    fontColor,
		    lineType)
	#print(rect1_to_ball)
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
