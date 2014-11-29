import rosbag
import math
from python_pointclouds import *
import csv

def main(fileName):

	bag = rosbag.Bag(fileName)

	msg_num = 0

	for topic, msg, t in bag.read_messages():
		
		if topic == "/head_mount_kinect/depth_registered/points":
			
			file_no_ext = fileName.rsplit(".")
			csv_name = file_no_ext[0] + "_msg_" + str(msg_num)
			print csv_name

			unpackMessage(msg, csv_name)

			msg_num += 1

			if msg_num >= 10:
				break

	bag.close()


def unpackMessage(msg, output_name):
	from std_msgs.msg import Int32, String

	rowStep = msg.row_step
	pointStep = msg.point_step
	height = msg.height
	width = msg.width

	# unpack the message to an array
	npArray = pointcloud2_to_array(msg)

	# the characteristics of the data
	numVals = 6
	xIndex = 0
	yIndex = 1
	zIndex = 2
	rIndex = 3
	gIndex = 4
	bIndex = 5

	vals_to_write = []
	i = 0

	# Create a list of tuple of (x,y,z,r,g,b) values from Kinect data
	for x in xrange(0, 640*480*numVals, numVals):

		zVal = npArray[i + zIndex]
		
		#if math.isnan(potX):
		#	potX = 0
		#if math.isnan(potY):
		#	potY = 0
		# Handle z nan's later so they can be correctly scaled for inverted greyscale image

		r = npArray[i + rIndex]
		g = npArray[i + gIndex]
		b = npArray[i + bIndex]
		
		vals_to_write.append(r)
		vals_to_write.append(g)
		vals_to_write.append(b)
		vals_to_write.append(zVal)

		i += numVals

	# write the data to a file
	f = open(output_name, 'w')
	csv_writer = csv.writer(f, delimiter = ',')
	csv_writer.writerow(vals_to_write)
	f.close()



if __name__ == '__main__':
	fileName = 'kinectKnobTest1.bag'
	main(fileName)

