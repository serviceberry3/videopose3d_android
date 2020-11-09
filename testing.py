import posenet
import cv2 as cv

image_src =  cv.imread("/home/nodog/Downloads/moad2.jpg")

posenet.estimate_pose(image_src)