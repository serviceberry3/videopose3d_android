# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Perform inference on a single video or all videos with a certain extension
(e.g., .mp4) in a folder.
"""

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor


import subprocess as sp
import numpy as np
import time
import argparse
import sys
import os
import glob
import posenet

from data_utils import suggest_metadata

def parse_args():
    #create an instance of ArgumentParser
    parser = argparse.ArgumentParser(description='End-to-end inference')

    #add expected arguments to the parser
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: mp4)',
        default='mp4',
        type=str
    )

    #passed input image or folder of images
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

#get resolution of the video
def get_resolution(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
    for line in pipe.stdout:
        w, h = line.decode().strip().split(',')
        return int(w), int(h)


def read_video(filename):
    #get resolution of video
    w, h = get_resolution(filename)

    command = ['ffmpeg',
            '-i', filename,
            '-f', 'image2pipe',
            '-pix_fmt', 'bgr24',
            '-vsync', '0',
            '-vcodec', 'rawvideo', '-']

    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)

    while True:
        data = pipe.stdout.read(w * h * 3)
        if not data:
            break
        yield np.frombuffer(data, dtype = 'uint8').reshape((h, w, 3))


def main(args):
    #if they passed a directory, get all images
    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)

    #otherwise just used the one passed image
    else:
        im_list = [args.im_or_folder]


    #gets some suggested COCO metadata like keypoint symmetries
    metadata = suggest_metadata('coco')

    '''NOTES
    {} defines an empty dict
    [] defines an empty list
    These are fundamentally different types. A dict is an associative array, a list is a standard array with integral indices.'''

    #set video metadata to empty dict
    metadata['video_metadata'] = {}

    #intialize final output dict to empty dict
    output = {}

    #for each video passed
    for video_name in im_list:
        #get absolute path of output directory
        #out_name = os.path.join(args.output_dir, os.path.basename(video_name))
        out_name=os.path.join(args.output_dir, "data_2d_custom_myvideos")

        #print name of video that we're processing
        print('Processing {}'.format(video_name))

        #intialize arrays to empty
        boxes = []
        segments = []
        keypoints = []

        #iterate over each frame in the vid
        for frame_i, im in enumerate(read_video(video_name)): #enumerate creates array of pairs--index paired with the video name
            #get the time
            t = time.time()

            #run the inference
            kps = posenet.estimate_pose(im)
            
            #print how long it took to process frame
            print('Frame {} processed in {:.3f}s'.format(frame_i, time.time() - t))


            has_bbox = False

            '''
            if kps.has('pred_boxes'):
                bbox_tensor = outputs.pred_boxes.tensor.numpy()
                if len(bbox_tensor) > 0:
                    has_bbox = True
                    scores = outputs.scores.numpy()[:, None]
                    bbox_tensor = np.concatenate((bbox_tensor, scores), axis=1)'''

            '''
            else:
                print("No keypts or boxes found in frame, setting keypts and boxes to empty array")
                kps = []
                bbox_tensor = []'''
            
            #add keypoints (excluding confidence checks) for this frame to array for whole video
            keypoints.append(kps[:, :2])

        
        #store video resolution in metadata
        this_vid_metadata = {
            'w': im.shape[1],
            'h': im.shape[0],
        }

        #get name of video we're processing
        canonical_name = video_name

        #intialize value keyed at [canonical_name] to empty dict
        output[canonical_name] = {}

        #add keypoint locations to output array
        output[canonical_name]['custom'] = [np.array(keypoints)]

        #add video resolution metadata to output array
        metadata['video_metadata'][canonical_name] = this_vid_metadata

        #print(keypoints)

        #export npz file containing all data retrieved by inference
        np.savez_compressed(out_name, positions_2d=output, metadata=metadata)


if __name__ == '__main__':
    setup_logger()

    #parse the passed arguments
    args = parse_args()

    #run the main fxn with args
    main(args)

