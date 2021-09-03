# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 10:03:25 2021

@author: hafizh
"""


from absl import flags
import sys
FLAGS = flags.FLAGS
sys.argv = sys.argv[:1]
FLAGS(sys.argv)

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from itertools import chain 

class_names = [c.strip() for c in open('data/labels/coco.names').readlines()]
yolo = YoloV3(classes=len(class_names))
yolo.load_weights('weights/yolov3.tf')

# Feature tracking
max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.8

model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)

vid = cv2.VideoCapture('data/video/plaza_cibubur_3.mp4')

# Mensetting format video output
codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps = int(vid.get(cv2.CAP_PROP_FPS))
vid_width, vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('data/video/results.avi', codec, vid_fps, (vid_width, vid_height))

# Perulangan untuk detection and tracking
while True:
  _, img = vid.read()
  if img is None:
      print('Completed')
      break

  img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert warna cv2 BGR ke format tersorflow RGB
  img_in = tf.expand_dims(img_in, 0) # Mengekspand dimensi
  img_in = transform_images(img_in, 416)

  t1 = time.time() # timer 

  boxes, scores, classes, nums = yolo.predict(img_in)

  # boxes, 3D shape (1, 100, 4) | 1 img = 100 Bounding box max | 4 = X,y,width,height
  # scores, 2D shape (1, 100) | confidence score
  # classes, 2D shape (1, 100) 
  # nums, 1D shape (1,) | total number detected objeck

  classes = classes[0]
  names = []
  for i in range(len(classes)):
      names.append(class_names[int(classes[i])])
  names = np.array(names)
  converted_boxes = convert_boxes(img, boxes[0])
  features = encoder(img, converted_boxes)

  detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                zip(converted_boxes, scores[0], names, features)]

  boxs = np.array([d.tlwh for d in detections])
  scores = np.array([d.confidence for d in detections])
  classes = np.array([d.class_name for d in detections])
  indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
  detections = [detections[i] for i in indices]

  tracker.predict()
  tracker.update(detections)

  cmap = plt.get_cmap('tab20b')
  colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]
  total_count = 0
  unsafe = []
  count = 0
  unsafe_id = []
  #LOOP TRACKER
  for track in tracker.tracks:
    if not track.is_confirmed() or track.time_since_update > 1:
      continue # jika "kalman filter" tidak update maka skip
    bbox = track.to_tlbr()
    class_name = track.get_class()
    color = colors[int(track.track_id) % len(colors)]
    color = [i * 255 for i in color]

    if class_name == 'person':
      total_count+=1
      centerixX = (bbox[0] + bbox[2]) // 2
      centerixY = (bbox[1] + bbox[3]) // 2

      cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 1)
      cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-18)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*8, int(bbox[1])), color, -1)
      cv2.putText(img, class_name+"-"+str(track.track_id), (int(bbox[0]+5), int(bbox[1]-5)), 0, 0.30,(255, 255, 255), 1)

    for tracky in tracker.tracks:
      if not tracky.is_confirmed() or tracky.time_since_update > 1:
        continue # jika "kalman filter" tidak update maka skip
      bboy = tracky.to_tlbr()
      class_name = tracky.get_class()
      if class_name == 'person':
        centeriyX = (bboy[0] + bboy[2]) // 2
        centeriyY = (bboy[1] + bboy[3]) // 2

        midlex = (centeriyX + centerixX) // 2
        midley = (centeriyY + centerixY) // 2

        distance = math.sqrt(math.pow(int(centeriyX)  - int(centerixX), 2) + math.pow(int(centeriyY) - int(centerixY), 2))

        if int(distance) <= 60:
          cv2.line(img, (int(centerixX), int(centerixY)), (int(centeriyX), int(centeriyY)), (255, 255, 255), 1)
          # count += 1count += 1
          if int(distance) != 0:
            cv2.circle(img,(int(centeriyX),int(centeriyY)), 2, (255,255,255), -1)
            cv2.putText(img, '{:0.0f}cm'.format(int(distance)), (int(midlex)-8, int(midley)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            unsafe.append([centeriyX, centeriyY])
            unsafe.append([centerixX, centerixY])
            
            cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (0,0,255), 2)
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-18)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*8, int(bbox[1])), (0,0,255), -1)
            cv2.putText(img, class_name+"-"+str(track.track_id), (int(bbox[0]+5), int(bbox[1]-5)), 0, 0.30,(255, 255, 255), 1)

    if centerixX in chain(*unsafe) and centerixY in chain(*unsafe):
        count += 1
        unsafe_id.append(track.track_id)
        # roi = img[int(bbox[3]):int(bbox[4]), int(bbox[0]):int(bbox[1])]
        # cv2.imwrite('read.jpg',roi)

    cv2.rectangle(img, (50, 50), (400, 100+70), (0, 0, 0), -1) # warna kotak title
    cv2.putText(img, 'people : {}'.format(total_count), (70, 80+5), 0, 0.9, (255, 255, 255), 1)
    cv2.putText(img, 'distance unsafe : < {}cm'.format('60'), (70, 80+40), 0, 0.6, (255, 255, 255), 1)
    cv2.putText(img, 'No. of people unsafe : {}'.format(count), (70, 80+70), 0, 0.6, (255, 255, 255), 1)

              
  fps = 1./(time.time()-t1)
  # cv2.putText(img, "FPS: {:.2f}". format(fps), (0,30), 0, 1, (0,0,255), 2)
  # cv2.resizeWindow('output', 1024, 768)
  cv2.imshow('output', img)

  out.write(img)

  if cv2.waitKey(1) == ord('q'):
    break

print(unsafe_id)
vid.release()
out.release()
cv2.destroyAllWindows()


