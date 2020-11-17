'''
Realtime 3D Human Reconstruction using Posenet and Facebook's VideoPose3D
3D drawing using pygtagrph based on OpenGL
Speed: TBD
'''
import os
import sys
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from pyqtgraph.opengl import *
import cv2

#progress bar animator
from tqdm import tqdm
import numpy as np
import time
import math

#Load the VideoPose3D model
from tools.utils import videopose_model_load as Model3Dload
model3D = Model3Dload()

#Load interface for running inference
from tools.utils import interface as interface3d

#Import more utils
from tools.utils import draw_3Dimg, draw_2Dimg, videoInfo, resize_img, common


#Import posenet 2D joint finder
from posenet import estimate_pose

common = common()

#initialize some global vars
item = 0
item_num = 0

#initialize pos_init to 17x3 zeros matrix
pos_init = np.zeros((17,3))


class Visualizer(object):
    def __init__(self):
        #initialize traces to blank dict
        self.traces = dict()


        self.app = QtGui.QApplication(sys.argv)
        self.w = gl.GLViewWidget()

        self.w.opts['distance'] = 45.0       #Distance of camera from center
        self.w.opts['fov'] = 60              #Horizontal field of view in degrees
        self.w.opts['elevation'] = 10       #Camera's angle of elevation in degrees
        self.w.opts['azimuth'] = 90         #Camera's azimuthal angle in degrees

        self.w.setWindowTitle('3D Visualization')
        self.w.setGeometry(450, 700, 980, 700) 
        self.w.show()

        #Create the background grids
        gx = gl.GLGridItem()
        gx.rotate(90, 0, 1, 0)
        gx.translate(-10, 0, 0)
        self.w.addItem(gx)
        gy = gl.GLGridItem()
        gy.rotate(90, 1, 0, 0)
        gy.translate(0, -10, 0)
        self.w.addItem(gy)
        gz = gl.GLGridItem()
        gz.translate(0, 0, -10)
        self.w.addItem(gz)

        #Special settings

        #Open up a VideoCapture for live frame feed
        self.cap = cv2.VideoCapture(0)

        #set video name
        #self.video_name = input_video.split('/')[-1].split('.')[0]

        #intialize 2D keypoints to empty array
        self.kpt2Ds = []

        pos = pos_init

        for j, j_parent in enumerate(common.skeleton_parents):
            if j_parent == -1:
                continue

            x = np.array([pos[j, 0], pos[j_parent, 0]]) * 10
            y = np.array([pos[j, 1], pos[j_parent, 1]]) * 10
            z = np.array([pos[j, 2], pos[j_parent, 2]]) * 10 - 10
            pos_total = np.vstack([x,y,z]).transpose()

            self.traces[j] = gl.GLLinePlotItem(pos=pos_total, color=pg.glColor((j, 10)), width=6,  antialias=True)
            self.w.addItem(self.traces[j])


    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            print("Starting qtgraph")
            #start up the qtgraph gui
            QtGui.QApplication.instance().exec_()


    def set_plotdata(self, name, points, color, width):
        self.traces[name].setData(pos=points, color=color, width=width)

    
    #Run the model on 30 frames at a time
    def update(self):
        #these globals get updated on every callback
        global item
        global item_num

        #set num to half of item
        num = item/2

        #calculate camera's current azimuthal angle in degrees and store it in the Visualizer item
        #azimuth_value = abs(num%120 + math.pow(-1, int((num/120))) * 120) % 120

        #self.w.opts['azimuth'] = azimuth_value

        #Log which frame this is
        #print("Frame #", item)


        #read in a frame from the VideoCapture (webcam)
        _, frame = self.cap.read() #ignore the other returned value
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        #update every other frame
        if item % 2 != 1:
            #resize the incoming image frame
            frame, W, H = resize_img(frame)

            #run Posenet inference to find the 2D joint keypoints
            joints_2D = estimate_pose(frame)

            #open pop-up and draw the keypoints found
            img2D  = draw_2Dimg(frame, joints_2D, 1)

            #if this is the first frame
            if item == 0:
                for _ in range(30):
                    self.kpt2Ds.append(joints_2D)

            else:
                self.kpt2Ds.append(joints_2D)
                self.kpt2Ds.pop(0)

            #increment the frame counter
            item += 1

            #run 2D-3D inference using VideoPose3D model
            joint3D = interface3d(model3D, np.array(self.kpt2Ds), W, H)

            #get the 3d coordinates
            pos = joint3D[-1] #(17, 3)


            for j, j_parent in enumerate(common.skeleton_parents):
                if j_parent == -1:
                    continue


                x = np.array([pos[j, 0], pos[j_parent, 0]]) * 10
                y = np.array([pos[j, 1], pos[j_parent, 1]]) * 10
                z = np.array([pos[j, 2], pos[j_parent, 2]]) * 10 - 10


                pos_total = np.vstack([x,y,z]).transpose()


                self.set_plotdata(name=j, points=pos_total, color=pg.glColor((j, 10)), width=6)


            d = self.w.renderToArray((img2D.shape[1], img2D.shape[0])) #(W, H)


            item_num += 1

        else:
            item += 1

    #Start up the live realtime 3D animation
    def animation(self):
        #instantiate a QTimer object to keep track of time during animation
        timer = QtCore.QTimer()

        #connect the "callback" fxn for timer timeout (to update drawing)
        timer.timeout.connect(self.update)

        #start the timer
        timer.start(1)

        #start the QApplication
        self.start()


def main():
    #Instantiate a Visualizer object for the input video file
    v = Visualizer()

    #Start up realtime 3D animation for Visualizer
    v.animation()

    #Close all open windows after animation ends
    cv2.destroyAllWindows()


#Main entrance point
if __name__ == '__main__':
    main()