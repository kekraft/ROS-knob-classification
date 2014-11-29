#!/usr/bin/env python

'''This is the main file for our term project.
    The project is to classify a few different set of knobs from point cloud data
    utilizing PyBrain.

    author: Kory Kraft
    date: 11/20/2014
    '''

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal

import numpy as np
import cv2

def main():

    #trial()
    ''' Read in the data '''
    ''' Massage data ''' # 640 x 480 
    ''' Build network '''
    ''' Train network '''

    ''' Give test data '''
    ''' output results '''
    #mine()

def trial():
    means = [(-1,0),(2,4),(3,1)]
    cov = [diag([1,1]), diag([0.5,1.2]), diag([1.5,0.7])]
    alldata = ClassificationDataSet(2, 1, nb_classes=3)
    #for n in xrange(400):
    # here we have random data created by selecting random samples from 3 diff guassian distributions
    for klass in range(3):
        samples = multivariate_normal(means[klass],cov[klass],400)
        for sample in samples:
            print sample
            #print [klass]
            alldata.addSample(sample, [klass])  # when adding a sample, why do you have to add a class with it?

    tstdata, trndata = alldata.splitWithProportion( 0.25 ) # 75 train, 25 test

    # advised to convert to having one ouput node
    trndata._convertToOneOfMany( )
    tstdata._convertToOneOfMany( )

    # testing dataset by printing out valuable info about it
    print "Number of training patterns: ", len(trndata)
    print "Input and output dimensions: ", trndata.indim, trndata.outdim
    print "First sample (input, target, class):"
    print trndata['input'][0], trndata['target'][0], trndata['class'][0]

    # building feed forward network #The output layer uses a softmax function because we are doing classification. 
    fnn = buildNetwork( trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer )

    # using backprop trainer here...
    trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

    # Now generate a square grid of data points and put it into a dataset, which we can then classify to obtain a nice contour field for visualization. 
    #Therefore the target values for this data set can be ignored.
    ticks = arange(-3.,6.,0.2)
    X, Y = meshgrid(ticks, ticks)
    # need column vectors in dataset, not arrays
    griddata = ClassificationDataSet(2,1, nb_classes=3)
    for i in xrange(X.size):
        griddata.addSample([X.ravel()[i],Y.ravel()[i]], [0])
    griddata._convertToOneOfMany()  # this is still needed to make the fnn feel comfy

    for i in range(20):
        trainer.trainEpochs( 5 )
        trnresult = percentError( trainer.testOnClassData(),
                              trndata['class'] )
        tstresult = percentError( trainer.testOnClassData(
               dataset=tstdata ), tstdata['class'] )

        print "epoch: %4d" % trainer.totalepochs, \
              "  train error: %5.2f%%" % trnresult, \
              "  test error: %5.2f%%" % tstresult


        out = fnn.activateOnDataset(griddata)
        out = out.argmax(axis=1)  # the highest output activation gives the class
        out = out.reshape(X.shape)

        # plot it as filled contour
        figure(1)
        ioff()  # interactive graphics off
        clf()   # clear the plot
        hold(True) # overplot on
        for c in [0,1,2]:
            here, _ = where(tstdata['class']==c)
            plot(tstdata['input'][here,0],tstdata['input'][here,1],'o')
        if out.max()!=out.min():  # safety check against flat field
            contourf(X, Y, out)   # plot the contour
        ion()   # interactive graphics on
        draw()  # update the plot


    ioff()
    show()


def mine():
    ''' Read in the data '''
    # for now create the data on my own
    #readData(fname)
    ''' Massage data ''' # 640 x 480 
    # load true image


    ''' Build network '''
    ''' Train network '''

    ''' Give test data '''
    ''' output results '''

def NN():
    rows = 640
    cols = 480
    numFeats = rows * cols
    classes = 10
    classLabels = ['knob1', 'knob2', 'knob3']
    hiddenNeurons = 60 # currently a random number ?
    numEpochs = 5
    numTrainings = 20

    #tstdata, trndata = alldata.splitWithProportion( 0.25 ) # 75 train, 25 test

    # Should we squash the data this point to be a vector? By the way, numpy.ravel simply flattens a list.
    trndata = ClassificationDataSet(numFeats, 1, nb_classes=classes, class_labels=classLabels)
    tstdata = ClassificationDataSet(numFeats, 1, nb_classes=classes, class_labels=classLabels)

    trndata._convertToOneOfMany()
    tstdata._convertToOneOfMany()

    # build the NN
    fnn = buildNetwork(trndata.indim, hiddenNeurons, trndata.outdim, outclass=SoftmaxLayer)

    # train the NN using backprop trainer here...
    trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

    for i in range(numTrainings):
        trainer.trainEpochs( numEpochs )
        trnresult = percentError( trainer.testOnClassData(),
                              trndata['class'] )
        tstresult = percentError( trainer.testOnClassData(
               dataset=tstdata ), tstdata['class'] )

        print "epoch: %4d" % trainer.totalepochs, \
              "  train error: %5.2f%%" % trnresult, \
              "  test error: %5.2f%%" % tstresult


        #out = fnn.activateOnDataset(griddata)
        #out = out.argmax(axis=1)  # the highest output activation gives the class
        #out = out.reshape(X.shape)

        # plot it as filled contour
        #figure(1)
        #ioff()  # interactive graphics off
        #clf()   # clear the plot
        #hold(True) # overplot on
        #for c in [0,1,2]:
        #     here, _ = where(tstdata['class']==c)
        #     plot(tstdata['input'][here,0],tstdata['input'][here,1],'o')
        # if out.max()!=out.min():  # safety check against flat field
        #     contourf(X, Y, out)   # plot the contour
        # ion()   # interactive graphics on
        # draw()  # update the plot


    # ioff()
    # show()
    '''
    # Load data from the given CSV files
    rawTraining = np.loadtxt(open("usps-4-9-train.csv","rb"),delimiter=",")
    # Get expected as a column vector. 
    # Note: first half of training examples expected val are 0
    #       second half of training examples expected val are 1
    expected = np.array(rawTraining[:,rawTraining.shape[1]-1]).T
    imgNrows = np.array(rawTraining[:,:rawTraining.shape[1]-1])
    '''
    # We will probably have an image in form of 

    '''
    x = zeros((rows, cols), dtype=uint8)  # Initialize numpy array
    y = zeros((cols, 1), dtype=uint8)  # Initialize numpy array
    for i in range(cols):
        if i % 1000 == 0:
            print("i: %i" % i)
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)  # Just a single byte
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                x[i][row][col] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]
    return (x, y)
    '''

def subSample(numpyArray, dimTuple):
    ''' Resizes/subSamples array to given size
        Returns: numpyArrayResized
        If image cannot be resized according to the dimensions supplied, 
            the None is returned
    '''
    # Test code
    #w,h = 512,512
    #data = np.zeros( (w,h,3), dtype=np.uint8)
    #dimTuple = (50,50)
    print 'Shape of array before', data.shape

    try:
        resize = cv2.resize(numpyArray, dimTuple)
        print 'Shape of array after resize: ', resize.shape
        return resize
    except:
        print 'Cannot resize array'
        return None


def averageArray(arr, n):
    ''' Takes a single row numpy array and averages over n elements
        Found on StackOverflow: http://stackoverflow.com/questions/10847660/subsampling-averaging-over-a-numpy-array
    ''' 
    end =  n * int(len(arr)/n)
    return numpy.mean(arr[:end].reshape(-1, n), 1)


if __name__ == '__main__':
    subSample(None)
    main()
    mine()
