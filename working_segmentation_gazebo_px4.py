#!/usr/bin/env python

import rospy
import cv2
import os
import sys
import random

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
#from PIL import Image

seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed

image_size = 128
train_path = "/home/bharathhh/drone_ws/src/train"
model=keras.models.load_model('model1.h5')

def func(frame):
#     img = Image.fromarray(frame)
#     img = img.resize((256, 256), Image.ANTIALIAS)
#     inp_numpy = numpy.array(img)[None]
#     inp = tf.constant(inp_numpy, dtype='float32')
    
    image = cv2.resize(frame, (image_size, image_size)).reshape(128,128,3)
#     image=image/255
    #cv2.imshow("image",image)
    segmentedImg = image.reshape(1,128,128,3)
    #cv2.imshow("segmented image",segmentedImg)
    segmentation_output = model.predict(segmentedImg)[0,:,:,0]

    # fig = plt.figure(figsize=(15,6))
    # ax = fig.add_subplot(2, 2)
    # ax.imshow(np.reshape((segmentation_output[0,:,:,0])*255, (image_size, image_size)), cmap="rgb")
    # cv2.imshow("123",segmentation_output)
    
    
    height = image.shape[0]
    width = image.shape[1]

    offset1 = 20
    offset2 = 5
    p1 = [height/2 - offset1,height/2]
    p2 = [height/2 + offset1,height/2]

    p3 = [height/2 + offset2,width-1]
    p4 = [height/2 - offset2,width-1]

    points = np.array([p1,p2,p3,p4])
    
    mask = np.zeros(segmentation_output.shape[0:2], dtype="uint8")

    cv2.fillPoly(mask, pts=np.int32([points]), color=(255, 255, 255))


#     image = cv2.resize(image, (image_size, image_size)).reshape(128,128,3)

#     segmentation_output = np.reshape(segmentation_output, (height, width))
    
    # performing a bitwise_and with the image and the mask
    maskedImg = cv2.bitwise_and(segmentation_output, segmentation_output, mask=mask) #After segmentation and then masking

#     print(segmentedImg.shape)
#    cv2.imshow('temp', segmentedImg)
    
#    segmentedImg = maskedImg.reshape(1,128,128,3)
    #cv2.imshow("segmentedImg", segmentedImg)
#     segmentedImg=segmentedImg/255
#     print(segmentedImg.shape)
#    segmentation_output = model.predict(segmentedImg)[0,:,:,0]
    #cv2.imshow("segmentation_output", segmentation_output)
    
 
    asd = np.uint8(plt.cm.gist_earth(maskedImg)*255)
    maskedImg = np.array(asd)
    #cv2.imshow("segmentation_output_2",segmentation_output)
    
    gray = cv2.cvtColor(maskedImg, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,150,255,0)
    #cv2.imshow("after gray scaling", gray)


    contours, hierarchies = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    blank = np.zeros(thresh.shape[:2],
                    dtype='uint8')
    
    cv2.drawContours(blank, contours, -1,
                    (255, 0, 0), 1)
    #cv2.imshow("contours",blank)
    maxarea = 0.0
    for i in contours:
        maxarea = max(cv2.contourArea(i), maxarea)
    #print(maxarea)   
    for i in contours:
        if cv2.contourArea(i)==maxarea:
            M = cv2.moments(i)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv2.drawContours(maskedImg, [i], -1, (0, 255, 0), 2)
                cv2.circle(maskedImg, (cx, cy), 4, (0, 0, 255), -1)
#                 print(f"x: {cx} y: {cy}")
    #cv2.imshow('temp', segmentation_output)
    cv2.imshow('temp1', maskedImg)

class camera_1:

  def __init__(self):
    self.image_sub = rospy.Subscriber("iris/usb_cam/image_raw", Image, self.callback)

  def callback(self,data):
    bridge = CvBridge()

    try:
      cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
    except CvBridgeError as e:
      rospy.logerr(e)
    
    image = cv_image
    image = cv2.resize(image, (image_size, image_size)).reshape(128,128,3)
    cv2.imshow("image",image)
    func(image)
    
    #resized_image = cv2.resize(image, (360, 640)) 

    #cv2.imshow("Camera output normal", image)
    #cv2.imshow("Camera output resized", image)

    cv2.waitKey(3)
  

def main():
  c=camera_1()

  try:
    rospy.spin()
  except KeyboardInterrupt:
    rospy.loginfo("Shutting down")
  
  cv2.destroyAllWindows()

if __name__ == '__main__':
    rospy.init_node('camera_read', anonymous=False)
    main()