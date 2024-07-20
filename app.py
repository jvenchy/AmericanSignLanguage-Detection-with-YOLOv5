import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import matplotlib

# Ensure matplotlib uses the appropriate backend
matplotlib.use('TkAgg')

###############################
# Testing Images with Yolo

# model = torch.hub.load('yolov5', 'yolov5s', source='local')
# img = 'traffic-jam-getty.jpg'
#
# results = model(img)
# results.print()
#
# # Render results
# rendered_img = np.squeeze(results.render())
#
# # Using matplotlib to display the image
# plt.imshow(rendered_img)
# plt.axis('off')  # Hide axis
# plt.show()
# plt.close()


###############################
# Real Time Detections with YOLO
#
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret, frame = cap.read()
#
#     # Make detection
#     results = model(frame)
#
#     cv2.imshow('YOLO', np.squeeze(results.render()))
#
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()


###############################
# Training the model (Use RectLabel on the MacOS App Store for labelling images taken)
# import uuid
# import os
# import time
#
# IMAGES_PATH = os.path.join('data', 'images')
# labels = ['Hello!', 'Yes!', 'No.', 'I Love You <3', 'Please...', 'Sorry']
# number_imgs = 20
#
# # Loop through labels
# cap = cv2.VideoCapture(0)
# for label in labels:
#     print('Collecting images for {}'.format(label))
#
#     # Loop through image range
#     for img_num in range(number_imgs):
#         print('Collecting images for {}, image number {}'.format(label, img_num))
#
#         # webcam feed
#         ret, frame = cap.read()
#
#         # name image path
#         imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
#
#         # write out image to file
#         cv2.imwrite(imgname, frame)
#
#         # render to screen
#         cv2.imshow('Image Collection', frame)
#
#         # 2 second delay between captures
#         time.sleep(2)
#
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

#####################################
# The actual training command:
# cd yolov5 && python train.py --img 320 --batch 16 --epochs 5 --data data/dataset.yaml --weights yolov5s.pt

######################################
# Load the trained model
trained_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp7/weights/best.pt')

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert frame to RGB (YOLOv5 expects RGB images)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Make detections
    results = trained_model(frame_rgb)

    # For Debugging: print results
    print(results.pandas().xyxy[0])  # Print the bounding box coordinates

    # Display results
    annotated_frame = np.squeeze(results.render())

    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('YOLO', annotated_frame_bgr)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
