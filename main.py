import math
import cv2
import cvzone
import numpy as np
import time
from ultralytics import YOLO
from sort import *
model= YOLO('./yolo-models/yolov8n.pt')
cap = cv2.VideoCapture('./videos/video1.mp4')
mask = cv2.imread('./images/mask.png')
def click_event(event,x,y,flag,params):
    print(x, ' ', y)
cap.set(3,1024)
cap.set(4,576)
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)
limits =[-50,285,1084,285]
lowerlimits=[-50,230,1084,230]
totalcount=[]
incoming_vehicle={}
outgoing_vehicle={}
while cap.isOpened():
     _ , frame = cap.read()
     frame_height, frame_width, _ = frame.shape
     mask = cv2.resize(mask, (frame_width, frame_height))
     masked_img = cv2.bitwise_and(frame,mask)
     if not _:
         break
     detections = np.empty((0,5))
     results = model(masked_img,stream=True)
     for r in results:
         boxes = r.boxes
         for box in boxes:
             if box.cls[0] == 2 or box.cls[0] == 7 or box.cls[0] == 3 or box.cls[0] == 5:
                 x1,y1,x2,y2 = box.xyxy[0]
                 x1,y1,x2,y2 = int(x1) , int(y1) , int(x2) , int(y2)
                 w, h = x2-x1 , y2-y1
                 cv2.circle(frame,(int(x1+w//2),int(y1+h//2)),4,(255,0,0),cv2.FILLED)
                 conf = math.ceil((box.conf[0]*100)) / 100
                 #cvzone.putTextRect(frame, 'vehicle', (max(0,x1),max(20,y1)),scale=1,offset=6, thickness=1,colorT=(0,0,0))
                 currentArray = np.array([x1,y1,x2,y2,conf])
                 detections = np.vstack((detections,currentArray))
     resultTracker = tracker.update(detections)
     cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), thickness=2)
     cv2.line(frame, (lowerlimits[0], lowerlimits[1]), (lowerlimits[2], lowerlimits[3]), (0, 0, 255), thickness=2)
     for result in resultTracker:
         x1,y1,x2,y2,id = result
         x1,y1,x2,y2 = int(x1) , int(y1) , int(x2),int(y2)
         w,h, = x2-x1 , y2-y1
         cvzone.cornerRect(frame,(x1,y1,w,h),l=9,rt=2,colorR=(255,0,0))
         #cvzone.putTextRect(frame, f'{int(id)}', (max(0, x1), max(35, y1)), scale=1, offset=3, thickness=2)
         x, y = x1 + w // 2, y1 + h // 2
         if limits[0] < x < limits[2] and limits[1] - 15 < y < limits[3] + 15 :
             if totalcount.count(id)==0:
                 totalcount.append(id)
                 outgoing_vehicle[id]=time.time()
                 cv2.line(frame, (int(limits[2]/2), limits[1]), (limits[2], limits[3]), (0, 255, 0), thickness=2)
             elif id in incoming_vehicle:
                 start = incoming_vehicle[id]
                 total_time = time.time()-start
                 speed_ms = 10/total_time
                 speed_km = speed_ms*25
                 cvzone.putTextRect(frame, f'{round(speed_km,2)}/hr', (max(0, x1), max(35, y1)), scale=1, offset=3, thickness=2)
         if lowerlimits[0] < x < lowerlimits[2] and lowerlimits[1] - 15 < y < lowerlimits[3] + 15:
             if totalcount.count(id)==0:
                 totalcount.append(id)
                 incoming_vehicle[id]=time.time()
                 cv2.line(frame, (lowerlimits[0], lowerlimits[1]), (int(lowerlimits[2]/2), lowerlimits[3]), (0, 255, 0), thickness=2)
             elif id in outgoing_vehicle:
                 start = outgoing_vehicle[id]
                 total_time = time.time()-start
                 speed_ms = 10/total_time
                 speed_km = speed_ms*25
                 cvzone.putTextRect(frame, f'{round(speed_km/2)}/hr', (max(0, x1), max(35, y1)), scale=1, offset=3, thickness=2)
         cvzone.putTextRect(frame, f' Totalcount: {len(totalcount)}, in:{len(incoming_vehicle)} , out:{len(outgoing_vehicle)}', (50, 50))
     cv2.imshow('frame', frame)
     #cv2.setMouseCallback('frame', click_event)
     cv2.waitKey(1)
cv2.destroyAllWindows()
cap.release()