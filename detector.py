import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

import torch
from tensorflow.keras.models import load_model
import mediapipe as mp
import pandas as pd

class Detector(object):
    """docstring for Detector"""
    def __init__(self):
        # Load Deepsort model :
          # Definition of the parameters
        print('load Deepsort model')

        max_cosine_distance = 0.05
        nn_budget = 100
        self.nms_max_overlap = 1.0
        
        # initialize deep sort
        model_filename = 'model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        print('encoder complete')
        # calculate cosine distance metric
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        # initialize tracker
        self.tracker = Tracker(metric, max_age=200)

        # initialze bounding box to empty
        self.bbox = ''
        self.frame_num = 0
        self.initialisation = True
        self.count = 0
        self.IDofInterest = -1

        self.count_redo_init = 0
        self.flaginfo=True

        # INITIALISE GESTURE RECOGNITION
        print('load hand gesture model')

        # Load the gesture recognizer model :
        self.model_gesture = load_model('hand_gesture-recognition-code/mp_hand_gesture')
        # Load class names
        
        f = open('hand_gesture-recognition-code/gesture.names', 'r')
        self.classNames = f.read().split('\n')
        f.close()
        print(self.classNames)
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_holistic = mp.solutions.holistic

        # Load YOLOv5 model :
        print('yolov5 model')

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.model.confidence = 0.4

        self.class_names = utils.read_class_names(cfg.YOLO.CLASSES)
    #def load(self, PATH):
        # self.model = torch.load(PATH)
        # self.model.eval()

        #self.model.load_state_dict(torch.load(PATH))
        #self.model.eval()

    def forward(self, frame):

        pred_bboxes = [80,60]
        pred_y_label = [1.0]
        frame = np.array(frame)
        frame_size = frame.shape[:2]
    
        with self.mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as holistic:
                    # Detection
            result = self.model(frame) # inference for person
            detectPerson = result.pandas().xyxy[0] # Get the bounding box, the confidence and the class
            detectPerson = detectPerson [(detectPerson['class']== 0)]
            if self.initialisation :
          
                num_objects = 0
                bboxes = np.array([])
                scores = np.array([])
                classes = np.array([])
                #className = 'okay'
                detectPersonOfInterest = pd.Series(dtype='float64')

                for i in range(detectPerson.shape[0]):
                    #print('detectPerson.index',detectPerson.index)

                    cur_person = detectPerson.iloc[i,:]
                    #print('cur_person', cur_person)
                    xmin = int(cur_person['xmin'])
                    xmax = int(cur_person['xmax'])
                    ymin = int(cur_person['ymin'])
                    ymax = int(cur_person['ymax'])
                    frame_crop = np.ascontiguousarray(frame[max(1,ymin-10):min(ymax+10,120),max(1,xmin-10):min(xmax+10,160),:])
                

                    frame_crop.flags.writeable = False 

                    # Make detection
                    results = holistic.process(frame_crop)
                    frame_crop.flags.writeable = True
    
                    # Extract landmarks 
                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
                        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

                        
                            #print('handmarks', handmarks)
                                        # Drawing landmarks on frames
                            #mpDraw.draw_landmarks(bbox_array[:,:,0:3], right_hand, mpHands.HAND_CONNECTIONS)
                       
                        #print('left_wrist',left_wrist.x)
                        #print('right_wrist',right_wrist.x)

                        #print('left_shouldert',left_shoulder.x)
                        #print('right_shoulder',right_shoulder.x)

                        if (left_wrist.y > left_shoulder.y) and (right_wrist.y < right_shoulder.y) and (left_wrist.x < left_shoulder.x) and (left_wrist.x > right_shoulder.x) and (right_wrist.x < left_shoulder.x) and (right_wrist.x > right_shoulder.x) :
                            detectPersonOfInterest = detectPerson.iloc[i,:] 
                            print('detect')
                            num_objects=1
                            bboxes = np.zeros((1,4))
                            scores = np.zeros(1)
                            classes = np.zeros(1)
                            bboxes[0,0] = int(detectPersonOfInterest['ymin'])/frame_size[0]
                            bboxes[0,1] = int(detectPersonOfInterest['xmin'])/frame_size[1]
                            bboxes[0,2] = int(detectPersonOfInterest['ymax'])/frame_size[0]
                            bboxes[0,3] = int(detectPersonOfInterest['xmax'])/frame_size[1]
                            scores[0] = np.array(float(detectPerson.iloc[0]['confidence']))
                            classes[0] = np.array(0) # class person
                            self.count += 1 
                            
                            if self.count > 10: 
                                self.initialisation = False
                                self.count = 0
                            break           

            elif not detectPerson.empty:
        # convert data to numpy arrays and slice out unused elements
          
                num_objects = detectPerson.shape[0]

                bboxes = np.zeros((num_objects,4))
                scores = np.zeros(num_objects)
                classes = np.zeros(num_objects)

                for i in range(num_objects):
                    #print('numobjects', num_objects)
                    #print('i',i)
                    bboxes[i,0] = int(detectPerson.iloc[i]['ymin'])/frame_size[0]
                    bboxes[i,1] = int(detectPerson.iloc[i]['xmin'])/frame_size[1]
                    bboxes[i,2] = int(detectPerson.iloc[i]['ymax'])/frame_size[0]
                    bboxes[i,3] = int(detectPerson.iloc[i]['xmax'])/frame_size[1]
                    scores[i] = np.array(float(detectPerson.iloc[i]['confidence']))
                    classes[i] = np.array(0) # class person

            else:
          
                num_objects = 0
                bboxes = np.array([])
                scores = np.array([])
                classes = np.array([])
            
            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(bboxes, original_h, original_w)

            # store all predictions in one parameter for simplicity when calling functions
            pred_bbox = [bboxes, scores, classes, num_objects]

            # loop through objects and use class index to get class name, allow only classes in allowed_classes list
            names = []
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = self.class_names[class_indx]
                names.append(class_name)


            # encode yolo detections and feed to tracker
            features = self.encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]
            
            #initialize color map
            cmap = plt.get_cmap('tab20b')
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            # run non-maxima supression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]       
            
            # Call the tracker
            self.tracker.predict()
            self.tracker.update(detections)

            # update tracks
            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()
                if self.initialisation :
                    if  results.pose_landmarks and not detectPersonOfInterest.empty:
                        self.IDofInterest = track.track_id
                    #print(IDofInterest)

                if track.track_id == self.IDofInterest :
                    self.count_redo_init = 0
                    pred_bboxes = [(bbox[0]+bbox[2])/(2),(bbox[1]+bbox[3])/(2)]  # check if it needs to be int() ,(bbox[2]-bbox[0]),(bbox[3]-bbox[1])
                    pred_y_label = [1.0]
                # if enable info flag then print details about each track
                if self.flaginfo:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
            #print('output bbox',pred_bboxes)
            #If the person of interest is lost, redo initialisation after 10 frames
            
            if not self.initialisation and  pred_bboxes[0] == 80:
                self.count_redo_init += 1
            if self.count_redo_init > 15:
                self.initialisation = True
                self.count_redo_init = 0
                print('Redo initialisation')

            #print('output',pred_y_label[0])
        return pred_bboxes, pred_y_label

"""detector = Detector()
from PIL import Image

# creating a object
#im = Image.open('image.jpg')
#im_array = np.array(im)
#detector.forward(im_array)
#pred_box,label = detector.forward(im_array)

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()

    frame = np.array(frame)
    
    #frame = cv2.resize(frame,(int(160),int(120)), interpolation=cv2.INTER_AREA)
    #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    
    frame = cv2.resize(frame,(int(160),int(120)), interpolation=cv2.INTER_AREA)
    factor=3
    #frame = cv2.resize(frame,(frame.shape[0]*factor,frame.shape[1]*factor), interpolation=cv2.INTER_AREA)
    
    pred_box,label = detector.forward(frame)
    cv2.rectangle(frame, (int(pred_box[0]-5), int(pred_box[1]-5)), (int(pred_box[0]+5), int(pred_box[1]+5)), [250,0,0], 2)
    #print(pred_box)
    cv2.imshow('Input', frame)
    c = cv2.waitKey(1)
    if c == 27:
       break

cap.release()
cv2.destroyAllWindows()
"""