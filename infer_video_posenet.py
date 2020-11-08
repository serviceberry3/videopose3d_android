import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import cv2 as cv
import math


#set path to model
path = "/home/nodog/Downloads/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"

#test image
img_path = "/home/nodog/Downloads/moad.jpg"

#name of pop-up
window_name = 'image'

#initialize interpreter
interpreter = tf.lite.Interpreter(model_path=path)

#allocate tensors for interpreter
interpreter.allocate_tensors()

#extract info about model’s input shape preferences and about the output to know which tensors to address later on
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#get required pixel dimensions of input image
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

#resize input image (testing right now on one image)
image_src = cv.imread(img_path)
# src_tepml_width, src_templ_height, _ = template_image_src.shape 
image_new = cv.resize(image_src, (width, height))
cv.imshow(window_name, image_new)

#waits for user to press any key  
cv.waitKey(0)  
  
#closing all open windows  
cv.destroyAllWindows() 


#add a new dimension to match model's input
input = np.expand_dims(image_new.copy(), axis=0)

#check the type of the input tensor
floating_model = input_details[0]['dtype'] == np.float32


#Floating point models offer the best accuracy, at the expense of model size 
#and performance. GPU acceleration requires use of floating point models.


#Model requires incoming imgs to have float values. Here we normalize them to be in range from 0 to 1 by subtracting
#default mean and dividing by default standard deviation.
if floating_model:
    input = (np.float32(input) - 127.5) / 127.5


# Process image
# Sets the value of the input tensor
interpreter.set_tensor(input_details[0]['index'], input)


# Runs the computation
interpreter.invoke()
# Extract output data from the interpreter
output_data = interpreter.get_tensor(output_details[0]['index'])
offset_data = interpreter.get_tensor(output_details[1]['index'])


# Getting rid of the extra dimension
heatmaps = np.squeeze(output_data)
offsets = np.squeeze(offset_data)

print("template_heatmaps' shape:", heatmaps.shape)
print("template_offsets' shape:", offsets.shape)

# The output consist of 2 parts:
# - heatmaps (9,9,17) - corresponds to the probability of appearance of 
# each keypoint in the particular part of the image (9,9)(without applying sigmoid 
# function). Is used to locate the approximate position of the joint
# - offset vectors (9,9,34) is called offset vectors. Is used for more exact
#  calculation of the keypoint's position. First 17 of the third dimension correspond
# to the x coordinates and the second 17 of them correspond to the y coordinates



