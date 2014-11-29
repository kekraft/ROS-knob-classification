from pybrain.datasets            import ClassificationDataSet
from pybrain.datasets import SupervisedDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pylab                       import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy                       import diag, arange, meshgrid, where
from numpy.random                import multivariate_normal

import numpy as np
import cv2
import csv
import math

def main():

	fileName1 = 'kinectKnobTest1_msg_0'
	fileName2 = 'kinectKnobTest1_msg_1'
	fileName3 = 'kinectKnobTest1_msg_2'
	fileName4 = 'kinectKnobTest1_msg_3'
	fileName5 = 'kinectKnobTest1_msg_4'

	nn_train_data = []
	nn_target = []

	# parse and append all data
	nn_train_data.append(parseData(fileName1))
	nn_train_data.append(parseData(fileName2))
	nn_train_data.append(parseData(fileName3))
	nn_train_data.append(parseData(fileName4))
	nn_target.append(1) # need to change when multiple classes
	nn_target.append(1) # need to change when multiple classes
	nn_target.append(1) # need to change when multiple classes
	nn_target.append(1) # need to change when multiple classes

	#print 'Length!', len(nn_train_data[0])
	#print nn_train_data[0]

	# create the dataset
	num_nn_inputs = len(nn_train_data[0])
	ds = SupervisedDataSet(num_nn_inputs, 1)

	for i in nn_train_data:
		ds.addSample((i), (1))

	for i in xrange(5):
		zero_vec = [0] * len(nn_train_data[0])
		ds.addSample((zero_vec), (0))

	print len(ds)
	print ds['input']
	print ds['target']

	# create the nn
	net = buildNetwork(num_nn_inputs, 10, 1) # arbitrary hidden cells for now

	# train
	trainer = BackpropTrainer(net, ds)
	trainer.trainEpochs( 250 )

	# test
	test_nn_pos = parseData(fileName5)
	test_nn_neg = [0] * len(nn_train_data[0])

	result_pos = net.activate(test_nn_pos)
	result_neg = net.activate(test_nn_neg)

	print "Should be 1: ", result_pos
	print "Should be 0: ", result_neg

def parseData(fileName):

	read_vals = readFile(fileName)

	# the characteristics of the file data: [R, G, B, Z]
	numVals = 4
	zIndex = 3
	rIndex = 0
	gIndex = 1
	bIndex = 2

	vals = []
	nn_vals = [] # for now just z-vals
	i = 0

	# Create a list of tuple of (x,y,z,r,g,b) values from Kinect data
	for x in xrange(0, 640*480*numVals, numVals):

		zVal = read_vals[i + zIndex]
		
		# Handle z nan's below so they can be correctly scaled for inverted greyscale image

		r = read_vals[i + rIndex]
		g = read_vals[i + gIndex]
		b = read_vals[i + bIndex]
		vals.append((r, g, b, zVal))
		nn_vals.append(zVal)
		
		i += numVals

	# reshape the list above into a list of lists of tubles
	#      the outer list holds all the rows
	#			the inner list holds the column vals for a particular row
	#				the tuple holds the (x,y,z,r,g,b) vals described above
	subsample_factor = 10

	imgStruct = []
	nn_inputs = []
	k = 0
	for i in range(480):
		row = []
		for j in range(640):
			if j%subsample_factor == 0:
				row.append(vals[k])
				nn_inputs.append(nn_vals[k])
			k +=1
		if i%subsample_factor == 0:
			imgStruct.append(row)

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

	# convert float to uint8 for display in opencv
	# get min and max of z value first
	# displayRGBListAsImg(zList)
	# Get the z value for a greyscale depth image, scaling each greyscale value between 0 and 255
	#   
	scaleFactor = (255.0 / maxZ)
	newZlist = [[[0 for k in xrange(3)] for j in xrange(640/subsample_factor)] for i in xrange(480/subsample_factor)]
	for i in range(480/subsample_factor):
		for j in range(640/subsample_factor):
			oldz = zList[i][j][0]
			if math.isnan(oldz):
				oldz = maxZ
			newZ = scaleFactor * oldz
			newZ = abs(newZ - 255) # invert color
			newZlist[i][j] = [newZ, newZ, newZ]

	# scale the z's for the nn_inputs
	out_nn_inputs = []

	for i in xrange(len(nn_inputs)):
		oldz = nn_inputs[i]
		if math.isnan(oldz):
			oldz = maxZ
		newZ = scaleFactor * oldz
		newZ = abs(newZ - 255) # inver color
		out_nn_inputs.append(newZ)
	
	#displayRGBListAsImg(rgbList)
	#displayRGBListAsImg(newZlist)

	return out_nn_inputs
	


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


def readFile(fileName):
	# read the data in from a csv file
	vals = []

	f = open(fileName, 'rb')
	reader = csv.reader(f)

	for row in reader:
		for col in row:
			vals.append(float(col))

	f.close()

	return vals

if __name__ == '__main__':
	main()
