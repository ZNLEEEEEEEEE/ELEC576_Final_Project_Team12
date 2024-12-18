#!/usr/bin/env python

''' dnn_detect.py

    Detect and draw target on frame based on the result of DNN.

'''

import cv2
import time

class dnn:
    def __init__(self,model_path):
        ''' Initialize DNN Model

            Args:
                model_path: The path of DNN model.
        '''
        # Initialize constants for DNN model
        self.CONFIDENCE_THRESHOLD = 0.2
        self.NMS_THRESHOLD = 0.4
        self.COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

        # Load DNN model
        self.class_names = []
        with open(model_path + "coco.names", "r") as f:
            self.class_names = [cname.strip() for cname in f.readlines()] 
        self.net = cv2.dnn.readNet(model_path + "yolov4-mish-416.weights", model_path + "yolov4-mish-416.cfg")
        
        # GPU accreleration
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1/255)
    
    def detect(self,frame):
        ''' Detect targets in current frame and return the properties of the target.

            Args:
                frame: Current frame that captured from the video or rosbag.

            Returns:
                frame: Current frame that captured from the video or rosbag.
                classes: The class names of all the detected targets. 
                scores: The scores of all the detected targets that outputs from the DNN model. 
                boxes: boxes: The pixel value of the detected box for all the the targets.
                start: Current frame start time.
                end: Current frame end time.
        '''
        start = time.time()
        classes, scores, boxes = self.model.detect(frame, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
        end = time.time()
        return frame,classes,scores,boxes,start,end

    def draw(self,frame,classes,scores,boxes,startTime,endTime):
        ''' Draw the boxes and scores of all the detected targets in current frame.

            Args:
                frame: Current frame that captured from the video or rosbag.
                classes: The class names of all the detected targets. 
                scores: The scores of all the detected targets that outputs from the DNN model. 
                boxes: boxes: The pixel value of the detected box for all the the targets.
                startTime: Current frame start time.
                endTime: Current frame end time.             
        '''
        start_drawing = time.time()
        for (classid, score, box) in zip(classes, scores, boxes):
            color = self.COLORS[int(classid) % len(self.COLORS)]
            label = "%s : %f" % (self.class_names[classid[0]], score)
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        end_drawing = time.time()
        fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (endTime - startTime), (end_drawing - start_drawing) * 1000)
        cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow("detections", frame)
    
    def get_class_names(self):
        ''' Return all the class name of the detected targets.

            Returns:
                self.class_names: All the class name of the detected targets.
        '''
        return self.class_names
