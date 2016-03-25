import cv2
import numpy
redUpper = (0,0,179)
redLower = (0,0, 81)
lighterImg = "./lighterImg.jpg"

def maskLighter(img):
    mask = cv2.inRange(img, redLower, redUpper)
    mask = cv2.erode(mask, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=3)
    return mask

def getImage():
	vidCap = cv2.VideoCapture(0)
	ret, img = vidCap.read()
	vidCap.release()
	return img

def showImage(img):
	cv2.imshow("Webcam", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def showVideo():
	vidCap = cv2.VideoCapture(0)
	watch = True
	while watch:
	    ret, img = vidCap.read()
	    mask = maskLighter(img)
	    cv2.imshow("Webcam", mask)
	    keyPress = cv2.waitKey(1)
	    if keyPress is not -1:
	    	watch = False
	cv2.destroyAllWindows()
	vidCap.release()

showVideo()