import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import cv2 as cv
import math


#set path to model
path = "/home/nodog/Downloads/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"

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

#resize input image

