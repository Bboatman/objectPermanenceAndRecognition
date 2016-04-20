import cv2

def getNewEntities(ogFrame, newFrame):
	grey = cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY)
	ogGrey = cv2.cvtColor(ogFrame, cv2.COLOR_BGR2GRAY)
	grey = cv2.GaussianBlur(grey, (51, 151), 0)
	ogGrey = cv2.GaussianBlur(ogGrey, (51, 151), 0)
	diff = cv2.absdiff(grey, ogGrey)
	threshImg = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)[1]
	threshImg = cv2.dilate(threshImg, None, iterations=5)
	cv2.imshow("Thresh", threshImg)
	cv2.imshow("Diff", diff)
	img, contours, h = cv2.findContours(threshImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	bounds = []
	for cont in contours:
		area = cv2.contourArea(cont)
		if area > 3000:
			bounds.append(cv2.boundingRect(cont))
	return bounds

camera = cv2.VideoCapture(0)
firstFrame = None
prevFrame = None
while True:
	success, frame = camera.read()
	if firstFrame is None:
		firstFrame = frame
	if not success:
		break
	bounds = getNewEntities(firstFrame, frame)
	for r in bounds:
		cv2.rectangle(frame, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (0,0,255))
	prevFrame = frame
	cv2.imshow("Video!", frame)
	key = cv2.waitKey(1)
	if key != -1:
		cv2.destroyAllWindows()
		break

