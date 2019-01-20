'''
This script show the code of yolo for object detection 
Remark: here we will use a existing pre-trained model 
		YOLO = You Only Look Once 

GHANMI Helmi 

20/01/2019

'''

#import required packages 
import sys, os 
import cv2
import matplotlib.pyplot as plt 
import numpy as np 

'''
1- define the threshold value of confidance and the non maxima supression 
2- define the width and the height of the image should be precessed 
3- Here we use 'coco' dataset so we get the label of classes from the file coco.names
4- load the confd yolo file yolov3.cfg
5- load the weight of the model yolov3.weights
6- define the dnn (deep neural network) backend from opencv 
7- reade the frame from video and apply the neural network 
8- detect, draw rectangle and put text ID of the detected object 

'''
#----------------------------------------------------------------------------------------------#

confThreshold = 0.3 # confidence threshould 
nmsThreshold= 0.5  # Non maxima supression threshold 

#with ans hight of the input image  
inputWidth =416
inputHight =416

# load the label of classe
ClasseFile = 'datasets.names'
classes = None

with open(ClasseFile, 'r') as f :
	#classes = len(f.readlines())
	classes = f.read().rstrip('\n').split('\n')
	print(classes)
	print('#-----------#')
	print('The number of classes is :{}'.format(len(classes)))

#----------------------------------------------------------------------------------------------#

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, such as the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#----------------------------------------------------------------------------------------------#


def postprocess(frame, outs,confThreshold, nmsThreshold):
	frameHeight = frame.shape[0]
	frameWidth = frame.shape[1]

	# deinfe the list of classid, confidences an boxes detection coordinates 
	classIDs = [] # all values of lcasses ID 
	confidences = [] # all values of confidences 
	boxes = [] # all coordiantes boxes for the  detection of many object in same frame 

	# the list of detection is [x, y, width, height, confidence, classID], where classID = 0,1,2,....80 (because coco dataset have 80 classes)
	for out in outs:
	    for detection in out:
	        scores = detection [5:]
	        classID = np.argmax(scores)
	        confidence = scores[classID]

	        if confidence > confThreshold:
	            centerX = int(detection[0] * frameWidth)
	            centerY = int(detection[1] * frameHeight)

	            width = int(detection[2]* frameWidth)
	            height = int(detection[3]*frameHeight )

	            left = int(centerX - width/2)
	            top = int(centerY - height/2)

	            classIDs.append(classID)
	            confidences.append(float(confidence))
	            boxes.append([left, top, width, height])

	indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold )
	for i in indices:
		i = i[0]
		box = boxes[i]
		left = box[0]
		top = box[1]
		width = box[2]
		height = box[3]
		
		drawPredObject(frame, classIDs[i], confidences[i], left, top, left+width, top+height)

	    #return confidences, classIDs, boxes

#----------------------------------------------------------------------------------------------#

# define the function to draw the rectangle around the detected object 
def drawPredObject(frame, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # draw a text with the label of the object 
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255),3)

#----------------------------------------------------------------------------------------------#
def run_yolo_detection(input_image):
	# load confidence and weights of yolo model 
	modelConf = 'yolov3.cfg'
	modelWeights = 'yolov3.weights'

	# apply the dnn (deep neural network) backend from opencv 
	net = cv2.dnn.readNetFromDarknet(modelConf, modelWeights)
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
	#print(type(net))

	# read input image 
	img = cv2.imread(input_image)

	# create a 4D blob from the img
	blob = cv2.dnn.blobFromImage(img, 1/255, (inputWidth, inputHight), [0,0,0], 1, crop = False)

	# set the blob image as in input of th first layer in the neural network(net)
	net.setInput(blob)

	# get the output of the net 
	outs = net.forward(getOutputsNames(net))

	# apply the funtion of postprocess 
	postprocess(img, outs,confThreshold,nmsThreshold)

	# save the detection results 
	cv2.imwrite('results/results_detection.png',img)
	# show the result 
	cv2.imshow('Object detection using YOLO',img)
	cv2.waitKey(3000)

	return img
#----------------------------------------------------------------------------------------------#


if __name__ == '__main__':

	input_image = 'input_test_image/image_test.jpg' 

	run_yolo_detection(input_image)

	print('Processing finished !')