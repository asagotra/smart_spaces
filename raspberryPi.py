######## Picamera Object Detection Using Tensorflow Classifier #########
#
# Author: Arun Sagotra
# Date: 3/06/2020

# Description:
"""
This program uses a TensorFlow2 classifier to perform object detection and count the people and appliances.
Also, calculates the person and appliances in each quadrant. It loads the classifier uses it to perform 
object detection on a Picamera feed and saved images. It draws boxes and scores around the objects of 
interest in each frame from the Picamera and saved images. 

"""
import os
import time
import sys
import Adafruit_DHT as dht
import paho.mqtt.client as mqtt
import json
import random
import numpy as np
import random, glob
import math 

## Import utilites in Raspberry Pi
#import picamera
from picamera.array import PiRGBArray
from picamera import PiCamera

from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from src.models import (YoloV3, YoloV3Tiny)
from src.dataset import transform_images, load_tfrecord_dataset
from src.utils import draw_outputs
from numpy import *
from collections import Counter
from datetime import datetime

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                        'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

app._run_init(['yolov3'], app.parse_flags_with_usage)



def get_subgrid_pos(detection_box):

    '''
    This function divide the objects in 2x2 grid 

    '''
    xmin, ymin, xmax, ymax = tuple(detection_box)
    if (ymin < 0.5) & (xmin < 0.5):
        grid_loc = 'Q2'
    elif (ymin < 0.5) & (xmin > 0.5):
        grid_loc = 'Q1'
    elif (ymin > 0.5) & (xmin < 0.5):
        grid_loc = 'Q3'
    elif (ymin > 0.5) & (xmin > 0.5):
        grid_loc = 'Q4'
    return grid_loc
 
# Window and Door Status  

def win_door_status(img1, img2):

    '''
    This function detect the status of door and window whether 
    its open or close

    '''
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    absdiff = cv2.absdiff(img1, img2)
    diff = cv2.subtract(img1, img2)
    result = not np.any(diff)
    if result:
        status = "Closed"
    else:
        status = "Open"
    return status

def cooling(current_temp, temp_setpoint):

    '''
    Function to determine if we need to switch the cooling on or off.

    '''
    if current_temp < temp_setpoint:
        hvac_status = 'Off'
    if current_temp >= temp_setpoint:
        hvac_status = 'On'
    return hvac_status

# Check if you need to connect to ThingsBoard
try:
    sys.argv[1]
    if (sys.argv[1] == 'connect'):
        connect_tb = True
    else:
        connect_tb = False
except:
    connect_tb = False

if connect_tb:
    THINGSBOARD_HOST = 'your host name'
    ACCESS_TOKEN = 'your device access token'
    
    attrib_data = {'firmware_version': '0', 'serial_number': '0'}
    client = mqtt.Client()
    # Set access token
    client.username_pw_set(ACCESS_TOKEN)
    # Connect to ThingsBoard using default MQTT port and 60 seconds keepalive interval
    client.connect(THINGSBOARD_HOST, 1883, 60)
    # Uploads firmware version and serial number as device attributes using 'v1/devices/me/attributes' MQTT topic
    attrib_data['firmware_version'] = '1.0.1'
    attrib_data['serial_number'] = 'SN-001'
    client.publish('v1/devices/me/attributes', json.dumps(attrib_data), 1)
    client.loop_start()

# Data capture and upload interval in seconds. Less interval will eventually hang the DHT22.
next_reading = time.time()
INTERVAL = 300
IM_WIDTH = 1920    
IM_HEIGHT = 1080   


try:
    while True:
        
        # Dictionary of Parameters to push into the thingboard 
        sensor_data = {'headCount': 0, 'q1_headcount': 0, 'q2_headcount': 0, 'q3_headcount': 0, 'q4_headcount': 0, 'appliances':0,
                        'temperature':-99, 'humidity':-99, 'doorstatus':'Open','windowstatus': 'Open', 'thermalload':0, 'q1_thermalload': 0,
                        'q2_thermalload': 0,'q3_thermalload': 0,'q4_thermalload': 0,'loadforecast': 0, 'energyconsumption': 0,'summary': 'Clear', 
                        'icon': 'clear-day', 'precipintensity': 0, 'precipprobability': 0, 'externaltemperature': 29.75,'apparenttemperature': 29.7,
                        'dewpoint': 13.24,'externalhumidity': 0.36, 'externalpressure': 1014.5, 'windspeed': 3.62, 'windgust': 3.94, 'windbearing': 143, 
                        'cloudcover': 0.19,'uvindex': 11, 'visibility': 16.093, 'ozone': 265.4, 'q1_temperature':-99, 'q2_temperature':-99,'q3_temperature':-99,
                        'q4_temperature':-99,'energysaving': 0, 'q1_appliances':0,'q2_appliances':0,'q3_appliances':0,'q4_appliances':0}
        

        camera_type = 'picamera'
        
        if camera_type=='picamera':
            camera = PiCamera()
            camera.resolution = (IM_WIDTH,IM_HEIGHT)
            time.sleep(0.1)
            
            camera.capture('data/image_pi.jpg')
            camera.close()
            
            print("Image captured!!!")

            physical_devices = tf.config.experimental.list_physical_devices('GPU')

            if len(physical_devices) > 0: 
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
        

            FLAGS.image = 'data/image_pi.jpg'
            
        elif camera_type == 'save_image':
            
            physical_devices = tf.config.experimental.list_physical_devices('GPU')

            if len(physical_devices) > 0: 
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
        
            filenames = glob.glob("data/*.jpg")
            filenames.sort()
            images = [img for img in filenames]

          #  FLAGS.image = 'data/image1.jpg'
            FLAGS.image = random.choice(images)

        if FLAGS.tiny:
            yolo = YoloV3Tiny(classes=FLAGS.num_classes)
        else:
            yolo = YoloV3(classes=FLAGS.num_classes)

        yolo.load_weights(FLAGS.weights).expect_partial()
        #logging.info('weights loaded')

        class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
        #logging.info('classes loaded')

        img_raw = tf.image.decode_image(
            open(FLAGS.image, 'rb').read(), channels=3)
        
        #print(img_raw)

        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, FLAGS.size)
        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        #logging.info('time: {}'.format(t2 - t1))


        #logging.info('detections:')
        
        i = 0
        objCounter = 0
        lapCounter = 0
        desCounter = 0
        sub_grid = True
        
        if sub_grid:
            obj_sub_grid = {'Q3':0, 'Q2':0, 'Q4':0, 'Q1':0}
            lap_sub_grid = {'Q3':0, 'Q2':0, 'Q4':0, 'Q1':0}
            des_sub_grid = {'Q3':0, 'Q2':0, 'Q4':0, 'Q1':0}
        else:
            obj_sub_grid = None

        # Person and Appliances Count

        for i in range(nums[0]):
            if class_names[int(classes[0][i])] =='person':
                objCounter += 1
                if sub_grid:
                    grid_key = get_subgrid_pos(np.array(boxes[0][i]))
                    obj_sub_grid[grid_key] += 1
            
            if class_names[int(classes[0][i])] =='laptop':
                lapCounter += 1
                if sub_grid:
                    grid_key = get_subgrid_pos(np.array(boxes[0][i]))
                    lap_sub_grid[grid_key] += 1
                
            if class_names[int(classes[0][i])] =='tvmonitor':
                desCounter += 1
                if sub_grid:
                    grid_key = get_subgrid_pos(np.array(boxes[0][i]))
                    des_sub_grid[grid_key] += 1
            
        i =i+1   
        
        appliances= lapCounter+desCounter
        
        appliances_subgrid = dict(lap_sub_grid)
        appliances_subgrid.update(des_sub_grid) 

        for i, j in lap_sub_grid.items():
            for x, y in des_sub_grid.items():
                if i == x:
                    appliances_subgrid[i]=(j+y)
                    
        
        img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        cv2.imwrite(FLAGS.output, img)
        #logging.info('output saved to: {}'.format(FLAGS.output))
        
        Q1temperature = round(random.uniform(20,28), 2)
        Q2temperature = round(random.uniform(20,28), 2)
        Q3temperature = round(random.uniform(20,28), 2)
        Q4temperature = round(random.uniform(20,28), 2)

        humidity,temperature = dht.read_retry(dht.DHT22, 4)
        
        imgd1 = './images/ref/your_ref_image.jpg'
        imgd2 = 'data/image_pi.jpg'
        
        imgw1 = './images/ref/your_ref_image.jpg'
        imgw2 = 'data/image_pi.jpg'
        door_status = win_door_status(imgd1, imgd2)
        window_status = win_door_status(imgw1, imgw2)
        
        # These values will come from thermal load model 
        q1_ThermalLoad = random.randint(50, 100)
        q2_ThermalLoad = random.randint(50, 100)
        q3_ThermalLoad = random.randint(50, 100)
        q4_ThermalLoad = random.randint(50, 100)
        thermalload = q1_ThermalLoad + q2_ThermalLoad + q3_ThermalLoad + q4_ThermalLoad
        LoadForecast = random.randint(100, 200)
        EnergyConsumption = random.randint(10, 100)
        energysaving = random.randint(0, 30)

        # hvac set temperature
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        start = '08:30:00'
        end = '18:00:00'
        if current_time >= start and current_time <= end:
            set_temp_cooling = 20
            
        else:
            set_temp_cooling = 23

        hvac_status_cooling = cooling(temperature, set_temp_cooling)
        

        # Data pushing to the thingsboard

        sensor_data['temperature'] = temperature
        sensor_data['q1_temperature'] = Q1temperature
        sensor_data['q2_temperature'] = Q2temperature
        sensor_data['q3_temperature'] = Q3temperature
        sensor_data['q4_temperature'] = Q4temperature
        sensor_data['humidity'] = humidity
        sensor_data['doorstatus'] = door_status
        sensor_data['windowstatus'] = window_status
        sensor_data['headCount'] = objCounter
        sensor_data['q3_headcount'] = obj_sub_grid['Q3']
        sensor_data['q4_headcount'] = obj_sub_grid['Q4']
        sensor_data['q2_headcount'] = obj_sub_grid['Q2']
        sensor_data['q1_headcount'] = obj_sub_grid['Q1']
        sensor_data['appliances'] = appliances
        sensor_data['q1_appliances'] = appliances_subgrid['Q1']
        sensor_data['q2_appliances'] = appliances_subgrid['Q2']
        sensor_data['q3_appliances'] = appliances_subgrid['Q3']
        sensor_data['q4_appliances'] = appliances_subgrid['Q4']
        sensor_data['thermalload'] = thermalload
        sensor_data['q1_thermalload'] = q1_ThermalLoad
        sensor_data['q2_thermalload'] = q2_ThermalLoad
        sensor_data['q3_thermalload'] = q3_ThermalLoad
        sensor_data['q4_thermalload'] = q4_ThermalLoad
        sensor_data['loadforecast'] = LoadForecast
        sensor_data['energyconsumption'] = EnergyConsumption
        sensor_data['energysaving'] = energysaving

        if connect_tb:
            client.publish('v1/devices/me/telemetry', json.dumps(sensor_data), 1)

        next_reading += INTERVAL
        sleep_time = next_reading-time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
            
except KeyboardInterrupt:
    pass


if connect_tb:
    client.loop_stop()
    client.disconnect()

