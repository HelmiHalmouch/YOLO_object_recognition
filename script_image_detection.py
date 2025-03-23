# -*- coding: utf-8 -*-

'''
This script demonstrates object detection using the YOLO (You Only Look Once) model.
Note: We are using an existing pre-trained YOLO model for this task.

Author: GHANMI Helmi
Date: 20/01/2019
'''

# Import required packages
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

class YoloObjectDetection:
    def __init__(self, model_cfg, model_weights, class_file, input_width=416, input_height=416, conf_threshold=0.3, nms_threshold=0.5):
        """
        Initializes the YOLO object detection class with necessary configurations and settings.
        :param model_cfg: Path to the YOLO configuration file (e.g., yolov3.cfg)
        :param model_weights: Path to the YOLO weights file (e.g., yolov3.weights)
        :param class_file: Path to the file containing class labels (e.g., coco.names)
        :param input_width: The input image width (default: 416)
        :param input_height: The input image height (default: 416)
        :param conf_threshold: Confidence threshold for object detection (default: 0.3)
        :param nms_threshold: Non-maxima suppression threshold (default: 0.5)
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_width = input_width
        self.input_height = input_height
        self.classes = self.load_classes(class_file)
        self.net = self.load_yolo_model(model_cfg, model_weights)

    def load_classes(self, class_file):
        """Load class labels from the specified file."""
        with open(class_file, 'r') as f:
            classes = f.read().rstrip('\n').split('\n')
            print(f"Loaded {len(classes)} classes")
        return classes

    def load_yolo_model(self, model_cfg, model_weights):
        """Load the YOLO model with the configuration and weights."""
        # Load YOLO model from configuration and weights files
        net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def get_outputs_names(self):
        """Get the names of all the output layers of the YOLO model."""
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        return output_layers

    def postprocess(self, frame, outs):
        """Process the outputs from YOLO and draw bounding boxes on the frame."""
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.conf_threshold:
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)

                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        for i in indices.flatten():
            box = boxes[i]
            left, top, width, height = box
            self.draw_prediction(frame, class_ids[i], confidences[i], left, top, left + width, top + height)

    def draw_prediction(self, frame, class_id, confidence, left, top, right, bottom):
        """Draw a bounding box and label on the image."""
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
        label = f"{self.classes[class_id]}: {confidence:.2f}"
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def run_detection(self, input_image):
        """Run YOLO object detection on an input image."""
        # Read the input image
        img = cv2.imread(input_image)

        # Create a blob from the image and set it as input to the network
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (self.input_width, self.input_height), [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)

        # Get the network outputs
        outs = self.net.forward(self.get_outputs_names())

        # Postprocess the output and draw bounding boxes on the frame
        self.postprocess(img, outs)

        # Save the result image
        cv2.imwrite('results/results_detection.png', img)

        # Show the result
        cv2.imshow('Object Detection using YOLO', img)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

        return img


if __name__ == '__main__':
    # Define file paths
    model_cfg = 'yolov3.cfg'
    model_weights = 'yolov3.weights'
    class_file = 'coco.names'
    input_image = 'input_test_image/image_test.jpg'

    # Initialize the YOLO detection model
    yolo_detector = YoloObjectDetection(model_cfg, model_weights, class_file)

    # Run detection on the input image
    yolo_detector.run_detection(input_image)

    print('Processing finished!')
