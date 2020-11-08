# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from glob import glob
import os
import sys

import argparse
from data_utils import suggest_metadata

#naming the output file
output_prefix_2d = 'data_2d_custom_'

def decode(filename):
    #Latin1 encoding because Detectron runs on Python 2.7
    print('Processing {}'.format(filename))

    #Load the npz file
    data = np.load(filename, encoding='latin1', allow_pickle=True)

    #get the different data types
    bb = data['boxes']
    kp = data['keypoints']
    metadata = data['metadata'].item()

    #initialize results arrays
    results_bb = []
    results_kp = []

    print("Len of bb is " + str(len(bb)))

    #iterate over all bounding boxes
    for i in range(len(bb)):
        #make sure we have some bounding boxes and keypoints
        if len(bb[i][1]) == 0 or len(kp[i][1]) == 0:
            print("No bbox/keypoints detected for this frame")
            #No bbox/keypoints detected for this frame -> will be interpolated
            results_bb.append(np.full(4, np.nan, dtype=np.float32)) # 4 bounding box coordinates
            results_kp.append(np.full((17, 4), np.nan, dtype=np.float32)) # 17 COCO keypoints
            continue

    
        '''
        bb is an array of (# frames) objects (e.g. this one has length 6), 
        each object containing: (1) an empty list, (2) an array containing one array of float32s.
        

        EXAMPLE bb:
        array(
            [
        
       [list([]),
        array([[3.3881702e+02, 1.7346634e+02, 7.1124298e+02, 1.4981326e+03, 
        9.9967420e-01]], dtype=float32)],

       [list([]),
        array([[3.3813235e+02, 1.7067661e+02, 7.1179724e+02, 1.4978040e+03,
        9.9966955e-01]], dtype=float32)],

       [list([]),
        array([[3.3865536e+02, 1.7064284e+02, 7.1188141e+02, 1.4969648e+03,
        9.9966598e-01]], dtype=float32)],

       [list([]),
        array([[3.4028592e+02, 1.6577415e+02, 7.1189526e+02, 1.4974039e+03,
        9.9964523e-01]], dtype=float32)],

       [list([]),
        array([[3.3993402e+02, 1.6699857e+02, 7.1184955e+02, 1.4974399e+03,
        9.9965644e-01]], dtype=float32)],

       [list([]),
        array([[3.9614020e+02, 1.5261166e+02, 7.6765546e+02, 1.4559672e+03,
        9.9978131e-01]], dtype=float32)]

        ], dtype=object

        )'''


        #find max of this bounding box's last float value. argmax run with just one number is always 0
        best_match = np.argmax(bb[i][1][:, 4]) #all rows, column 4. 

        #best_match is always 0 in our case

        '''
        formatted_float = "{:.2f}".format(best_match)
        print(formatted_float)
        '''

        #this will be the first 4 floats since best_match always 0
        best_bb = bb[i][1][best_match, :4] #ARRAY OF 4 FLOATS

        '''
        kp is an array of (#frames) objects, each obj containing: (1) an empty list, 
        (2) an array containing an array containing four arrays of float32s.

        EXAMPLE kp:
        array(
            [
       [list([]),
        array([
            [[5.1208862e+02, 5.3797144e+02, 4.8332980e+02, 5.7248199e+02,
         4.5169516e+02, 6.3862708e+02, 3.9992938e+02, 6.7745148e+02,
         3.7979825e+02, 6.7169971e+02, 3.7836032e+02, 6.0986835e+02,
         4.6032278e+02, 6.0411658e+02, 4.6176074e+02, 5.8686139e+02,
         4.6319864e+02],
        [2.9369421e+02, 2.7065652e+02, 2.7065652e+02, 2.9081448e+02,
         2.9513403e+02, 4.6503687e+02, 4.7943542e+02, 6.6805640e+02,
         6.8101508e+02, 8.5811719e+02, 8.7827521e+02, 8.6099689e+02,
         8.6531647e+02, 1.1302496e+03, 1.1417686e+03, 1.3793445e+03,
         1.4052620e+03],
        [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
         0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
         0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
         0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
         0.0000000e+00],
        [1.9178991e+00, 2.4641764e+00, 1.8484390e+00, 1.7734872e+00,
         1.0886503e+00, 2.4858075e-01, 4.3658778e-01, 7.3468459e-01,
         5.3171885e-01, 8.2581621e-01, 5.6063050e-01, 1.4108638e-01,
         1.2655501e-01, 4.4857776e-01, 5.9280270e-01, 4.0977404e-01,
         4.0157521e-01]]
         ], dtype=float32)],
       [list([]),
        array([[[5.1131165e+02, 5.3861792e+02, 4.8400537e+02, 5.7311011e+02,
         4.5095038e+02, 6.3922003e+02, 4.0064935e+02, 6.7802368e+02,
         3.7909174e+02, 6.7227496e+02, 3.7765460e+02, 6.1047656e+02,
         4.5957343e+02, 6.0329065e+02, 4.6244778e+02, 5.8604462e+02,
         4.6388498e+02],
        [2.9230594e+02, 2.6927554e+02, 2.7071494e+02, 2.8942715e+02,
         2.9518475e+02, 4.6359464e+02, 4.7798865e+02, 6.6798950e+02,
         6.8094409e+02, 8.5799042e+02, 8.7670264e+02, 8.6086920e+02,
         8.6662677e+02, 1.1329159e+03, 1.1415522e+03, 1.3790535e+03,
         1.4035232e+03],
        [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
         0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
         0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
         0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
         0.0000000e+00],
        [1.3576705e+00, 1.6736017e+00, 1.2638817e+00, 1.8019165e+00,
         8.7972760e-01, 2.4363431e-01, 4.5828825e-01, 7.5856525e-01,
         5.2261370e-01, 8.0828941e-01, 5.8094239e-01, 1.4034374e-01,
         1.2457452e-01, 4.5069775e-01, 6.2100667e-01, 4.2118776e-01,
         4.1337422e-01]]], dtype=float32)],
       [list([]),
        array([[[5.1163129e+02, 5.3890552e+02, 4.8435709e+02, 5.7335718e+02,
         4.5134091e+02, 6.3938940e+02, 3.9966348e+02, 6.7814752e+02,
         3.7956665e+02, 6.7240558e+02, 3.7813120e+02, 6.1067975e+02,
         4.5995380e+02, 6.0350232e+02, 4.6282480e+02, 5.8627649e+02,
         4.6426028e+02],
        [2.9233038e+02, 2.6928897e+02, 2.7072903e+02, 2.8945020e+02,
         2.9521057e+02, 4.6370099e+02, 4.7810187e+02, 6.6819360e+02,
         6.8115442e+02, 8.5828546e+02, 8.7700659e+02, 8.5972546e+02,
         8.7556647e+02, 1.1347826e+03, 1.1405428e+03, 1.3781576e+03,
         1.4040791e+03],
        [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
         0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
         0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
         0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00,
         0.0000000e+00],
        [1.3601316e+00, 1.6517327e+00, 1.1990516e+00, 1.8463761e+00,
         8.4460777e-01, 2.4371733e-01, 4.5716539e-01, 7.5630307e-01,
         5.3895235e-01, 8.0623585e-01, 5.6219959e-01, 1.3926953e-01,
         1.2568229e-01, 4.5887944e-01, 6.2420577e-01, 4.2464101e-01,
         3.9417556e-01]]], dtype=float32)],
'''

        #this will be a copy of transpose of the 2D array (4x17) of floats (kp[i][1][0] is the 2D array from the 3D)
        best_kp = kp[i][1][best_match].T.copy() #i is which bounding box iteration we're on
        #2D ARRAY OF FLOATS (17 rows, 4 columns since transposed)

        #append the array of 4 floats for this bounding box (probly upper left lower rt coords or something) to results_bb
        results_bb.append(best_bb)

        #append 17x4 2D array of floats for this keypoints (probly represents keypoint coords) to results_kp.append
        results_kp.append(best_kp)
        
    #create np arrays of results_bb and results_kp
    bb = np.array(results_bb, dtype=np.float32)
    #bb is now a 2D array (array of float arrays)

    kp = np.array(results_kp, dtype=np.float32)

    #kp is now a 3D array (array of 2D arrays)

    '''
    EXAMPLE:
    array([
        [[5.1208862e+02, 2.9369421e+02, 0.0000000e+00, 1.9178991e+00],
        [5.3797144e+02, 2.7065652e+02, 0.0000000e+00, 2.4641764e+00],
        [4.8332980e+02, 2.7065652e+02, 0.0000000e+00, 1.8484390e+00],
        [5.7248199e+02, 2.9081448e+02, 0.0000000e+00, 1.7734872e+00],
        [4.5169516e+02, 2.9513403e+02, 0.0000000e+00, 1.0886503e+00],
        [6.3862708e+02, 4.6503687e+02, 0.0000000e+00, 2.4858075e-01],
        [3.9992938e+02, 4.7943542e+02, 0.0000000e+00, 4.3658778e-01],
        [6.7745148e+02, 6.6805640e+02, 0.0000000e+00, 7.3468459e-01],
        [3.7979825e+02, 6.8101508e+02, 0.0000000e+00, 5.3171885e-01],
        [6.7169971e+02, 8.5811719e+02, 0.0000000e+00, 8.2581621e-01],
        [3.7836032e+02, 8.7827521e+02, 0.0000000e+00, 5.6063050e-01],
        [6.0986835e+02, 8.6099689e+02, 0.0000000e+00, 1.4108638e-01],
        [4.6032278e+02, 8.6531647e+02, 0.0000000e+00, 1.2655501e-01],
        [6.0411658e+02, 1.1302496e+03, 0.0000000e+00, 4.4857776e-01],
        [4.6176074e+02, 1.1417686e+03, 0.0000000e+00, 5.9280270e-01],
        [5.8686139e+02, 1.3793445e+03, 0.0000000e+00, 4.0977404e-01],
        [4.6319864e+02, 1.4052620e+03, 0.0000000e+00, 4.0157521e-01]]
        ],
      dtype=float32)'''

    kp = kp[:, :, :2] #Extract (x, y). Select just first two columns of all 2D arrays
    #kp is now basically an array of 4x2 float array

    '''EXAMPLE:
    array([
        [[ 512.0886 ,  293.6942 ],
        [ 537.97144,  270.65652],
        [ 483.3298 ,  270.65652],
        [ 572.482  ,  290.81448],
        [ 451.69516,  295.13403],
        [ 638.6271 ,  465.03687],
        [ 399.92938,  479.43542],
        [ 677.4515 ,  668.0564 ],
        [ 379.79825,  681.0151 ],
        [ 671.6997 ,  858.1172 ],
        [ 378.36032,  878.2752 ],
        [ 609.86835,  860.9969 ],
        [ 460.32278,  865.31647],
        [ 604.1166 , 1130.2496 ],
        [ 461.76074, 1141.7686 ],
        [ 586.8614 , 1379.3445 ],
        [ 463.19864, 1405.262  ]]
        ], dtype=float32)
    '''
    
    #IGNORE FOR NOW
    #Fix missing bboxes/keypoints by linear interpolation


    mask = ~np.isnan(bb[:, 0]) #all rows, first column


    indices = np.arange(len(bb))


    for i in range(4):
        #go through each column for all bounding box coords and do interpolation
        bb[:, i] = np.interp(indices, indices[mask], bb[mask, i])

    #go through all keypoint coords and do interpolation
    for i in range(17):
        for j in range(2):
            kp[:, i, j] = np.interp(indices, indices[mask], kp[mask, i, j])


    #logging
    print('{} total frames processed'.format(len(bb)))
    print('{} frames were interpolated'.format(np.sum(~mask))) #always coming up 0, don't worry about this
    print('----------')
    
    #return the data package
    return [{
        'start_frame': 0, # Inclusive
        'end_frame': len(kp), # Exclusive
        'bounding_boxes': bb,
        'keypoints': kp,
    }], metadata


if __name__ == '__main__':
    #make sure we're in right directory
    if os.path.basename(os.getcwd()) != 'data':
        print('This script must be launched from the "data" directory')
        exit(0)
        
    parser = argparse.ArgumentParser(description='Custom dataset creator')
    parser.add_argument('-i', '--input', type=str, default='', metavar='PATH', help='detections directory')
    parser.add_argument('-o', '--output', type=str, default='', metavar='PATH', help='output suffix for 2D detections')
    args = parser.parse_args()
    
    if not args.input:
        print('Please specify the input directory')
        exit(0)
        
    if not args.output:
        print('Please specify an output suffix (e.g. detectron_pt_coco)')
        exit(0)
    
    print('Parsing 2D detections from', args.input)
    
    metadata = suggest_metadata('coco')
    metadata['video_metadata'] = {}
    
    output = {}

    #get all npz files (all video outputs)
    file_list = glob(args.input + '/*.npz')

    #iterate over all npz files found
    for f in file_list:
        canonical_name = os.path.splitext(os.path.basename(f))[0]

        #decode the file to get our data (the keypoint locations) and metadata
        data, video_metadata = decode(f)

        output[canonical_name] = {}

        #add keypoint locations to output array
        output[canonical_name]['custom'] = [data[0]['keypoints'].astype('float32')]

        #add metadata to output array
        metadata['video_metadata'][canonical_name] = video_metadata

    #create new npz with output data
    print('Saving...')
    np.savez_compressed(output_prefix_2d + args.output, positions_2d=output, metadata=metadata)
    print('Done.')