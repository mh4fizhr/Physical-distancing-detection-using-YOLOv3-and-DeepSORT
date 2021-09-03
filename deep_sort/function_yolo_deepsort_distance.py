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

class yolo_deepsort_distance:

     def __init__(self,video_path):
          self.class_names = [c.strip() for c in open('data/labels/coco.names').readlines()]
          self.yolo = YoloV3(classes=len(class_names)
          self.yolo.load_weights('weights/yolov3.tf')

          # Feature tracking
          self.max_cosine_distance = 0.5
          self.nn_budget = None
          self.nms_max_overlap = 0.8

          self.model_filename = 'model_data/mars-small128.pb'
          self.encoder = gdet.create_box_encoder(self.model_filename, batch_size=1)
          self.metric = nn_matching.NearestNeighborDistanceMetric('cosine', self.max_cosine_distance, self.nn_budget)
          self.tracker = Tracker(self.metric)

          self.vid = cv2.VideoCapture(video_path)

          # Mensetting format video output
          self.codec = cv2.VideoWriter_fourcc(*'XVID')
          self.vid_fps = int(vid.get(cv2.CAP_PROP_FPS))
          self.vid_width, self.vid_height = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
          self.out = cv2.VideoWriter('data/video/results.avi', codec, vid_fps, (self.vid_width, self.vid_height))

          self.yolo()

          self.vid.release()
          out.release()
          cv2.destroyAllWindows()
          
     
     def yolo(self):
          while True:
               _, self.img = vid.read()
               if self.img is None:
                    print('Completed')
                    break

               self.img_in = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB) # convert warna cv2 BGR ke format tersorflow RGB
               self.img_in = tf.expand_dims(self.img_in, 0) # Mengekspand dimensi
               self.img_in = transform_images(self.img_in, 416)

               self.t1 = time.time() # timer 

               self.boxes, self.scores, self.classes, self.nums = yolo.predict(self.img_in)

               # boxes, 3D shape (1, 100, 4) | 1 img = 100 Bounding box max | 4 = X,y,width,height
               # scores, 2D shape (1, 100) | confidence score
               # classes, 2D shape (1, 100) 
               # nums, 1D shape (1,) | total number detected objeck


               self.deepsort()


               # self.fps = 1./(time.time()-t1)
               # cv2.putText(img, "FPS: {:.2f}". format(fps), (0,30), 0, 1, (0,0,255), 2)
               # cv2.resizeWindow('output', 1024, 768)
               cv2.imshow('output', self.img)

               out.write(self.img)

               if cv2.waitKey(1) == ord('q'):
               break
          

     def deepsort(self):
          self.classes = classes[0]
          self.names = []
          for i in range(len(self.classes)):
               self.names.append(class_names[int(classes[i])])
          self.names = np.array(self.names)
          self.converted_boxes = convert_boxes(self.img, boxes[0])
          self.features = encoder(self.img, converted_boxes)

          self.detections = [Detection(self.bbox, self.score, self.class_name, self.feature) for bbox, score, class_name, feature in
                         zip(converted_boxes, scores[0], names, features)]

          self.boxs = np.array([d.tlwh for d in self.detections])
          self.scores = np.array([d.confidence for d in self.detections])
          self.classes = np.array([d.class_name for d in self.detections])
          self.indices = preprocessing.non_max_suppression(self.boxs, self.classes, self.nms_max_overlap, self.scores)
          self.detections = [self.detections[i] for i in self.indices]

          tracker.predict()
          tracker.update(detections)

          self.cmap = plt.get_cmap('tab20b')
          self.colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]
          self.total_count = 0
          self.unsafe = []
          self.count = 0
          self.unsafe_id = []
          #LOOP TRACKER
          for track in tracker.tracks:
          if not track.is_confirmed() or track.time_since_update > 1:
               continue # jika "kalman filter" tidak update maka skip
          self.bbox = track.to_tlbr()
          self.class_name = track.get_class()
          self.color = self.colors[int(track.track_id) % len(self.colors)]
          self.color = [i * 255 for i in color]

          if self.class_name == 'person':
               self.total_count+=1
               self.centerixX = (self.bbox[0] + self.bbox[2]) // 2
               self.centerixY = (self.bbox[1] + self.bbox[3]) // 2

               cv2.rectangle(self.img, (int(self.bbox[0]),int(self.bbox[1])), (int(self.bbox[2]),int(self.bbox[3])), color, 1)
               cv2.rectangle(self.img, (int(self.bbox[0]), int(self.bbox[1]-18)), (int(self.bbox[0])+(len(self.class_name)+len(str(track.track_id)))*8, int(self.bbox[1])), color, -1)
               cv2.putText(self.img, self.class_name+"-"+str(track.track_id), (int(self.bbox[0]+5), int(self.bbox[1]-5)), 0, 0.30,(255, 255, 255), 1)

          # for tracky in tracker.tracks:
          #      if not tracky.is_confirmed() or tracky.time_since_update > 1:
          #      continue # jika "kalman filter" tidak update maka skip
          #      bboy = tracky.to_tlbr()
          #      class_name = tracky.get_class()
          #      if class_name == 'person':
          #      centeriyX = (bboy[0] + bboy[2]) // 2
          #      centeriyY = (bboy[1] + bboy[3]) // 2

          #      midlex = (centeriyX + centerixX) // 2
          #      midley = (centeriyY + centerixY) // 2

          #      distance = math.sqrt(math.pow(int(centeriyX)  - int(centerixX), 2) + math.pow(int(centeriyY) - int(centerixY), 2))

          #      if int(distance) <= 60:
          #           cv2.line(img, (int(centerixX), int(centerixY)), (int(centeriyX), int(centeriyY)), (255, 255, 255), 1)
          #           # count += 1count += 1
          #           if int(distance) != 0:
          #           cv2.circle(img,(int(centeriyX),int(centeriyY)), 2, (255,255,255), -1)
          #           cv2.putText(img, '{:0.0f}cm'.format(int(distance)), (int(midlex)-8, int(midley)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
          #           unsafe.append([centeriyX, centeriyY])
          #           unsafe.append([centerixX, centerixY])
                    
          #           cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (0,0,255), 2)
          #           cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-18)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*8, int(bbox[1])), (0,0,255), -1)
          #           cv2.putText(img, class_name+"-"+str(track.track_id), (int(bbox[0]+5), int(bbox[1]-5)), 0, 0.30,(255, 255, 255), 1)

          # if centerixX in chain(*unsafe) and centerixY in chain(*unsafe):
          #      count += 1
          #      unsafe_id.append(track.track_id)

          # cv2.rectangle(img, (50, 50), (400, 100+30), (0, 0, 0), -1) # warna kotak title
          # cv2.putText(img, 'people : {}'.format(total_count), (70, 80), 0, 0.7, (255, 255, 255), 1)
          # cv2.putText(img, 'No. of people unsafe: {}'.format(count), (70, 80+30), 0, 0.7, (255, 255, 255), 1)


          
     