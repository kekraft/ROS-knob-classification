import rosbag
import math
from python_pointclouds import *

def main(file):
	j = 0
	for topic, msg, t in bag.read_messages():
		
		if topic == "/head_mount_kinect/depth_registered/points":
			# print "Time: ", t
			# print "Point Cloud Dimensions (HxW): ", msg.height, msg.width
			# print "Row_Step: ", msg.row_step
			# print "Point_Step: ", msg.point_step
			# print "Fields: ", msg.fields
			# print "Data Value: ", msg.data[1]
			if j == 0:
				print 'unpacking in method'
				unpackMessage(msg)
				j = 1

	#140     int rgb = *reinterpret_cast<int*>(&rgb_data);
	#141     scalar[0] = ((rgb >> 16) & 0xff);
	#142     scalar[1] = ((rgb >> 8) & 0xff);
	#143     scalar[2] = (rgb & 0xff); 
	bag.close()


def unpackMessage(msg):
	from std_msgs.msg import Int32, String
	# string()
	#print msg
	
	# print String(msg)
	rowStep = msg.row_step
	pointStep = msg.point_step
	height = msg.height
	width = msg.width

	# for field in msg.fields:
	# 	#print field

	#print len(msg.data), rowStep * height
	#print type(msg.data)

	#print 'Height: ', height
	#print 'Width: ', width

	npArray = pointcloud2_to_array(msg)

	#print npArray
	#print npArray.shape[0]/(640*480)
	#print type(npArray[0])
	#print type(npArray[1])
	#print type(npArray[2])
	#print type(npArray[3])
	#print npArray[0]
	#print npArray[1]
	#print npArray[2]
	#print npArray[3]

	# working code if x y z and rgb
	# vals = []
	# i = 0
	# for x in xrange(0, 640*480*4, 4):
	# 	xVal = npArray[i]
	# 	yVal = npArray[i + 1]
	# 	zVal = npArray[i + 2]
	# 	rgb = npArray[i + 3]
	# 	vals.append((xVal, yVal, zVal, rgb))
	# 	i += 4

	# print 'len of vals: ', len(vals)
	# i = 0
	# for val in vals:
	# 	if i%30000== 0:
	# 		print 'Vals:{0}, ({1:10.6f},{2:10.6f},{3:10.6f}) , rgb-{4}'.format(i, val[0], val[1], val[2], val[3])
	# 	i +=1

	numVals = 6
	xIndex = 0
	yIndex = 1
	zIndex = 2
	rIndex = 3
	gIndex = 4
	bIndex = 5

	vals = []
	i = 0
	# Create a list of tuple of (x,y,z,r,g,b) values from Kinect data
	for x in xrange(0, 640*480*numVals, numVals):
		potX = npArray[i + xIndex]
		potY = npArray[i + yIndex]
		potZ = npArray[i + zIndex]
		
		if math.isnan(potX):
			#print 'x is not a num: ', potX
			potX = 0
		if math.isnan(potY):
			potY = 0
		# Handle z nan's below so they can be correctly scaled for inverted greyscale image
		# if math.isnan(potZ):
		# 	potZ = 0

		xVal = potX
		yVal = potY
		zVal = potZ
		r = npArray[i + rIndex]
		g = npArray[i + gIndex]
		b = npArray[i + bIndex]
		vals.append((xVal, yVal, zVal, r, g, b))
		i += numVals

	# reshape the list above into a list of lists of tubles
	#      the outer list holds all the rows
	#			the inner list holds the column vals for a particular row
	#				the tuple holds the (x,y,z,r,g,b) vals described above
	imgStruct = []
	k = 0
	for i in range(480):
		row = []
		for j in range(640):
			row.append(vals[k])
			k +=1
		imgStruct.append(row)
	print 'num rows ',len(imgStruct)

	# Get a list of lists of lists for img display:)
	# We want each element of the "matrix" to be a list of rgb values
	maxZ = 0
	minZ = 10000
	rgbList = []
	zList = []
	for row in imgStruct:
		rgbRow = []
		zRow = []
		for colVal in row:
			r = colVal[rIndex] 
			g = colVal[gIndex]
			b = colVal[bIndex]
			z = colVal[zIndex]

			if z > maxZ:
				maxZ = z
			if z < minZ:
				minZ = z
			rgb = [r, g, b]
			zs = [z, z, z]
			rgbRow.append(rgb)
			zRow.append(zs)
		rgbList.append(rgbRow)
		zList.append(zRow)

	print 'Max z ' , maxZ
	print 'Min z ' , minZ
	# convert float to uint8 for display in opencv
	# get min and max of z value first
	# displayRGBListAsImg(zList)
	# Get the z value for a greyscale depth image, scaling each greyscale value between 0 and 255
	#   
	scaleFactor = (255.0 / maxZ)
	newZlist = [[[0 for k in xrange(3)] for j in xrange(640)] for i in xrange(480)]
	for i in range(480):
		for j in range(640):
			oldz = zList[i][j][0]
			if math.isnan(oldz):
				oldz = maxZ
			newZ = scaleFactor * oldz
			newZ = abs(newZ - 255)
			newZlist[i][j] = [newZ, newZ, newZ]
	

	displayRGBListAsImg(rgbList)
	displayRGBListAsImg(newZlist)
		

	# print out 10 examples from the data
	print 'len of vals: ', len(vals)
	i = 0
	for val in vals:
		if i%30000== 0:
			print 'Vals:{0}, ({1:10.6f},{2:10.6f},{3:10.6f}) , ({4:3},{5:3},{6:3})'.format(i, val[0], val[1], val[2], val[3], val[4], val[5])
		i +=1

	pass


def displayRGBListAsImg(rgbList):
	''' Takes an RGB list and displays it as an image.
		Thus the list must similar to a format of [ [ [rgb1] [rgb2]  ] , [ [rgb3] [rgb4] ]  ] 
	'''

	# Need to be bgr!!!!!
	import numpy as np
	import cv2

	print "OpenCV Version: " + cv2.__version__

	# convert list to np array
	rgb = np.asarray(rgbList, dtype = np.uint8)

	# # create empty image
	# rgb = np.zeros((5, 10, 3), dtype = np.uint8)
	# rgb[1,1] = [255, 0, 0]
	# rgb[2,2] = [0, 255, 0]

	# test resizing
	'''
	image = np.zeros((5,10))

	resized = cv2.resize(image, (50, 100))

	cv2.imshow("output", image)
	cv2.waitKey(0)

	cv2.imshow("output", resized)
	cv2.waitKey(0)
	'''

	# display image
	cv2.namedWindow("output", flags = cv2.WINDOW_NORMAL)

	cv2.imshow("output", rgb)
	cv2.waitKey(0)


if __name__ == '__main__':
	bag = rosbag.Bag('kinectKnobTest1.bag')
	main(bag)

