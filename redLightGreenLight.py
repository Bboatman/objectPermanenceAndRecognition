import cv2
import numpy as np
from sklearn import cluster
from scipy import spatial
import random

def findEntities(firstFrame, prevFrame, newFrame):
	# Cast all imgs to grey
	grey = cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY)
	firstGrey = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
	prevGrey = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)

	# Blur and Subtract
	grey = cv2.GaussianBlur(grey, (51, 121), 0)
	firstGrey = cv2.GaussianBlur(firstGrey, (51, 121), 0)
	prevGrey = cv2.GaussianBlur(prevGrey, (51, 121), 0)
	blend = cv2.addWeighted(firstGrey, .8, prevGrey, .2, 5)
	diff = cv2.absdiff(grey, blend)

	# Threshold and Dialate
	threshImg = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
	threshImg = cv2.dilate(threshImg, None, iterations=2)

	# Show images (for debugging could be commented out)
	cv2.imshow("Thresh", threshImg)
	cv2.imshow("Diff", diff)

	# Find points of interest
	img, contours, h = cv2.findContours(threshImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours 
	bounds = []
	for cont in contours:
		area = cv2.contourArea(cont)
		bounds.append(cv2.boundingRect(cont))
	bounds = entityDetector(contours, bounds, newFrame)
	return bounds
	
def entityDetector(contours, boundingRects, img):
	# Check for cluster breaking state
	if len(boundingRects) <= 1:
		return boundingRects

	# Keep track of center points of all contours
	cropArr = []
	for rectangle in boundingRects:
		crop = img[rectangle[1]: rectangle[1]+rectangle[3], rectangle[0]: rectangle[0] + rectangle[2]]
		hist,bins = np.histogram(crop.ravel(),256,[0,256])
		histList = hist.tolist()
		histList.append(rectangle[0] + rectangle[2] // 2)
		histList.append(rectangle[1] + rectangle[3] // 2)
		cropArr.append(histList)

	# Set up clustering
	pref = [3 for x in range(256)]
	pref.append(5)
	pref.append(5)
	km = cluster.AffinityPropagation(damping=.8, preference=pref)
	clusterIndices = km.fit_predict(cropArr)
	entityArray = [[10000000,10000000,0,0] for x in range(max(clusterIndices) + 1)]

	# Find outer bounding box of cluster's contours
	for i in range(len(clusterIndices)):
		index = clusterIndices[i]
		rect = boundingRects[i]
		h = rect[3]
		w = rect[2]
		entityArray[index][0] = rect[0] if rect[0] < entityArray[index][0] else entityArray[index][0]
		entityArray[index][1] = rect[1] if rect[1] < entityArray[index][1] else entityArray[index][1]
		entityArray[index][2] = w if w > entityArray[index][2] else entityArray[index][2]
		entityArray[index][3] = h if h > entityArray[index][3] else entityArray[index][3]

	return entityArray

userList = []
camera = cv2.VideoCapture(0)
firstFrame = None
prevFrame = None
while True:
	success, frame = camera.read()
	if firstFrame is None:
		firstFrame = frame
	if prevFrame is None:
		prevFrame = frame
	if not success:
		break

	bounds = getNewEntities(firstFrame, prevFrame, frame)

	if len(bounds) > 0:
		for bound in bounds:
			crop = frame[bound[1]: bound[1]+bound[3], bound[0]: bound[0] + bound[2]]
			hist,bins = np.histogram(crop.ravel(),256,[0,256])

			# Create first user
			if len(userList) == 0:
				newhistList = hist.tolist()
				addcolor = (random.randint(0, 255), random.randint(0,255), random.randint(0,255))
				userList.append((newhistList,addcolor))

			matched = False
			# Iterate through possible users and decide which one this is
			for user in userList:
				compHist = user[0]
				testHist = hist.tolist()
				result = 1 - spatial.distance.cosine(compHist, testHist)
				if result > .5:
					color = user[1]
					matched = True
					break

			# If user hasn't been seen, add to list
			if not matched:
				histList = hist.tolist()
				color = (random.randint(0, 255), random.randint(0,255), random.randint(0,255))
				userList.append((histList,color))
			rectangle = cv2.rectangle(frame, (bound[0], bound[1]), (bound[0] + bound[2], bound[1] + bound[3]), color)

	# Display live entity tracking and destroy on keypress
	cv2.imshow("Video!", frame)
	prevFrame = frame
	key = cv2.waitKey(1)
	if key != -1:
		cv2.destroyAllWindows()
		break
