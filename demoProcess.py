'''
Authors: Brooke Boatman, Waab Hermes, Taylor Keppler
Date: May 2016
Demo File showing all pre and post processing.
Press 'a' to cycle through phases and any other 
key to exit
'''

import cv2
import numpy as np
from sklearn import cluster
from scipy import spatial
from random import triangular
from random import randint
from time import clock
from copy import deepcopy

breakpt = 500
imgArr = []


class User(object):
	'''
	User object for remembering clusters of points of interest 
	and detecting gross object movement
	'''
	def __init__(self, histogram):
		self.histogram = histogram
		self.isRelevant = False
		self.isOut = False
		self.currentRectangles = []
		self.massCenter = (0,0)
		self.prevCenter = (0,0)
		self.color = (randint(0, 255), randint(0,255), randint(0,255))

	def updateCenterMass(self):
		'''
		Keep track of a cluster of points of interest's center of mass
		'''
		self.prevCenter = self.massCenter
		x = 0
		y = 0
		for rect in self.currentRectangles:
			x += (rect[0] + (rect[2]/2)) / len(self.currentRectangles)
			y += (rect[1] + (rect[3]/2)) / len(self.currentRectangles)
		self.massCenter = (x,y)
		self.currentRectangles = []

	def didMove(self):
		''' 
		Calculate the distance between center of mass and previous 
		center as motion detection.
		'''
		self.updateCenterMass()
		totalDist = np.sqrt((self.massCenter[0] - self.prevCenter[0])**2 + (self.massCenter[1] - self.prevCenter[1])**2)
		if totalDist > 20:
			return True
		return False


def findEntities(firstFrame, prevFrame, newFrame):
	''' Locate points of interest for futher analysis'''
	# Cast all imgs to grey
	grey = cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY)
	firstGrey = cv2.cvtColor(firstFrame, cv2.COLOR_BGR2GRAY)
	prevGrey = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)

	# Blur and Subtract
	grey = cv2.GaussianBlur(grey, (11, 51), 0)
	firstGrey = cv2.GaussianBlur(firstGrey, (11, 51), 0)
	prevGrey = cv2.GaussianBlur(prevGrey, (11, 51), 0)
	blend = cv2.addWeighted(firstGrey, .8, prevGrey, .2, 5)
	diff = cv2.absdiff(grey, blend)

	# Threshold and Dialate
	threshImg = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
	threshImg = cv2.dilate(threshImg, None, iterations=2)

	# Show images for debugging
	video = deepcopy(newFrame)
	imgArr.append(video)
	imgArr.append(blend)
	imgArr.append(diff)
	imgArr.append(threshImg)

	threshCopy = deepcopy(threshImg)

	# Find points of interest
	img, contours, h = cv2.findContours(threshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	imgArr.append(threshCopy)
	bounds = []
	for cont in contours:
		area = cv2.contourArea(cont)
		bounds.append(cv2.boundingRect(cont))
	bounds = entityDetector(bounds, newFrame)
	return bounds

def grabSample(img, boundingRect):
	''' Grab a crop from the whole frame, and calculate it's histogram '''
	crop = img[boundingRect[1]: boundingRect[1]+boundingRect[3], boundingRect[0]: boundingRect[0] + boundingRect[2]]
	hist,bins = np.histogram(crop.ravel(),256,[0,256])
	return crop, hist.tolist()
	
def entityDetector(boundingRects, img):
	'''
	Grab cropping from every located bound
	'''
	# Check for cluster breaking state, if only one pt, can't cluster it
	if len(boundingRects) <= 1:
		return boundingRects

	# Keep track of center points of all contours
	cropArr = []
	for rectangle in boundingRects:
		crop, histList = grabSample(img, rectangle)
		cropArr.append(histList)

	bounds = clusterRelatedContours(cropArr, boundingRects)
	return bounds
	

def clusterRelatedContours(cropArr, boundingRects):
	'''
	Cluster croppings and combine them into one mass cropping
	'''
	# Set up clustering
	pref = [3 for x in range(256)]
	pref.append(5)
	pref.append(5)
	prev = None

	km = cluster.Birch(threshold=.5)
	clusterIndices = km.fit_predict(cropArr)
	entityArray = [[10000000,10000000,0,0] for x in range(max(clusterIndices) + 1)]
	# Find outer bounding box of all of a cluster's contours
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

def getPlayer(playerList, histogram):
	''' Match a cropping to a player '''
	# Create first user
	if len(playerList) == 0:
		player = User(histogram)
		playerList.append(player)
		return playerList, 0

	matched = False
	# Iterate through possible users and decide which one this is
	userId = len(playerList)
	for player in playerList:
		compHist = player.histogram
		result = 1 - spatial.distance.cosine(compHist, histogram)
		if result > .85:
			i = playerList.index(player)
			avgHist = [sum(x)/len(x) for x in zip(compHist, histogram)]
			playerList[i].histogram = avgHist
			matched = True
			userId = i
			break

	# If user hasn't been seen, add to list
	if not matched:
		playerList.append(User(histogram))
	return playerList, userId

def getImgSlice(frame, centerPt):
	'''
	Get a vertical slice of the image around a central point
	'''
	shapeTuple = frame.shape
	height = shapeTuple[0]
	width = shapeTuple[1]
	left = (centerPt[0] - 100) if (centerPt[0] - 100) > 0 else 0
	right = (centerPt[0] + 100) if (centerPt[0] + 100) < width else width
	sliceImg = frame[0:height, left:right]
	return sliceImg


def frameSubtractionDetection(prevFrame, newFrame, centerPt):
	'''
	Secondary motion detection algorithm
	'''
	grey = cv2.cvtColor(newFrame, cv2.COLOR_BGR2GRAY)
	prevGrey = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)

	# Blur and Subtract
	grey = cv2.GaussianBlur(grey, (11, 51), 0)
	prevGrey = cv2.GaussianBlur(prevGrey, (11, 51), 0)

	diff = cv2.absdiff(grey, prevGrey)
	sliceImg = getImgSlice(diff, centerPt)
	sumVals = np.sum(sliceImg)/(255*len(sliceImg))
	if sumVals >= 1:
		return True
	return False

def checkRelevance(user):
	'''
	See if a detected user is actually a relevent person
	'''
	user.isRelevant = False
	if user.isOut:
		user.isRelevant = False
	# If multiple clusters are being detected, probably a player		
	elif len(user.currentRectangles) > 1:
		user.isRelevant = True
	# If comprised of one very large cluster, probably a player
	elif len(user.currentRectangles) == 1:
		if user.currentRectangles[0][2] > breakpt or user.currentRectangles[0][3] > breakpt:
			user.isRelevant = True
	return user.isRelevant

def main():
	'''
	The main game loop
	'''
	global imgArr
	# Intialize
	userList = []
	camera = cv2.VideoCapture(0)
	firstFrame = None
	prevFrame = None
	currentIndex = 0

	while True:
		success, frame = camera.read()
		if firstFrame is None:
			firstFrame = frame
			prevFrame = frame

		# If you can't get camera, don't even try
		if not success:
			break

		bounds = findEntities(firstFrame, prevFrame, frame)
		if len(bounds) > 0:
			for bound in bounds:
				crop, hist = grabSample(frame, bound)
				userList, ID = getPlayer(userList, hist)
				userList[ID].currentRectangles.append(bound)
				# Debugging rectangle draw
				if userList[ID].isRelevant:
						rectangle = cv2.rectangle(frame, (bound[0], bound[1]), (bound[0] + bound[2], bound[1] + bound[3]), userList[ID].color)

		for user in userList:
			if checkRelevance(user):
				inMotion = user.didMove() or frameSubtractionDetection(frame, prevFrame, user.massCenter)
				# Debugging center point draw
				color = (0,0, 255) if inMotion else (0,255,0)
				cv2.circle(frame, user.massCenter, 5, color, 2)
			
		# Display live entity tracking and destroy on keypress
		imgArr.append(frame)
		prevFrame = frame
		key = cv2.waitKey(1)
		if key == 1048673:
			currentIndex += 1
			if currentIndex >= len(imgArr):
				currentIndex = 0
		elif key != -1:
			cv2.destroyAllWindows()
			break
			
		cv2.imshow('Motion Tracking' , imgArr[currentIndex])
		imgArr = []

main()
