# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno

from common.camera import *
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common.utils import deterministic_random

args = parse_args()
print(args)

try:
    #Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

#get path to dataset
print('Loading dataset...')
dataset_path = 'data/data_3d_' + args.dataset + '.npz'

#don't use ellipsis to truncate arrays when printing
#np.set_printoptions(threshold=sys.maxsize)

#choose dataset
if args.dataset == 'h36m':
    from common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path)
elif args.dataset.startswith('humaneva'):
    from common.humaneva_dataset import HumanEvaDataset
    dataset = HumanEvaDataset(dataset_path)
elif args.dataset.startswith('custom'):
    print("CUSTOM DATASET")
    from common.custom_dataset import CustomDataset

    #check path of npz
    print('PATH: outs/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')

    #create new CustomDataset object
    dataset = CustomDataset('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz') #NOTE CHANGE
else:
    raise KeyError('Invalid dataset')

print('Preparing data...')

print(dataset.subjects()) #looks like dict_keys(['../vids/output.mp4'])
for subject in dataset.subjects(): #should have just one subject, which will be '../vids/output.mp4'
    print(dataset[subject]) 
    '''looks like {'custom': {'cameras': {'id': '../vids/output.mp4', 'res_w': 1080, 'res_h': 1920, 
    'azimuth': 70, 'orientation': array([ 0.14070565, -0.15007018, -0.7552408 ,  0.62232804], dtype=float32), 
    'translation': array([1.841107 , 4.9552846, 1.5634454], dtype=float32)}}}'''

    print(dataset[subject].keys()) #something like dict_keys(['custom'])

    #should just be one key 'custom'
    for action in dataset[subject].keys():
        anim = dataset[subject][action]
        print(anim) 
        
        '''anim looks like {'cameras': {'id': '../vids/output.mp4', 'res_w': 1080, 'res_h': 1920, 'azimuth': 70, 
        'orientation': array([ 0.14070565, -0.15007018, -0.7552408 ,  0.62232804], dtype=float32), 'translation': array([1.841107 , 
        4.9552846, 1.5634454], dtype=float32)}}'''
        
        if 'positions' in anim:
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])

                pos_3d[:, 1:] -= pos_3d[:, :1] #Remove global offset, but keep trajectory in first position

                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d
        else:
            print("No positions in anim")

print('Loading 2D detections...')

#LOAD THE 2D DATA

#load the output of prepare_data_2d_custom.py
keypoints = np.load('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True) #NOTE CHANGE

#get the metadata
keypoints_metadata = keypoints['metadata'].item()

#get keypoints symmetry
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']

#separate keypoints symmetry lists
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])

#get joints from the h3.6m skeleton
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())

#get actual keypoint coords
keypoints = keypoints['positions_2d'].item()


#DO CHECKS (DOESN'T APPLY TO US)
for subject in dataset.subjects(): #should have just one subject, which will be '../vids/output.mp4'
    #make sure this video title is in keypoints
    assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)

    #should just be one key 'custom'
    for action in dataset[subject].keys():
        assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)


        if 'positions_3d' not in dataset[subject][action]: #USUALLY NOT
            print("positions_3d not in dataset['../vids/output.mp4]['custom']")
            continue
        


        for cam_idx in range(len(keypoints[subject][action])):
            
            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
            assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
            
            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

        assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])


print(dataset.cameras())
        

#NORMALIZE THE KEYPOINTS FOR THE RES
for subject in keypoints.keys(): #should just be one subject if one video (i.e. '../vids/output.mp4')
    print("SUBJECT: ", subject)
    for action in keypoints[subject]: #should just be one action, which is 'custom'
        print("ACTION: ", action)

        for cam_idx, kps in enumerate(keypoints[subject][action]): #each kps should be the Fx17x2 array of coordinates, where F is num frames
            print("INDEX ", cam_idx, "KPS: ", kps)


            print(dataset.cameras()[subject][cam_idx])
            '''looks like {'id': '../vids/output.mp4', 'res_w': 1080, 'res_h': 1920, 'azimuth': 70, 'orientation': array([ 0.14070565, 
            -0.15007018, -0.7552408 ,  0.62232804], dtype=float32), 'translation': array([1.841107 , 4.9552846, 1.5634454], dtype=float32)}'''

            #Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]

            #modify the keypoints. Replace them with res of running normalize_screen_coords with them for the given camera res
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])

            print("KEYPOINTS AFTER NORM_SCREEN_COORDS: ", kps)

            #store new kps in actual keypoints
            keypoints[subject][action][cam_idx] = kps


#SEMI-SUPERVISED TRAINING CHECK
subjects_train = args.subjects_train.split(',')
print("SUBJECTS_TRAIN", subjects_train) #looks like ['S1', 'S5', 'S6', 'S7', 'S8']

subjects_semi = [] if not args.subjects_unlabeled else args.subjects_unlabeled.split(',')

if not args.render:
    subjects_test = args.subjects_test.split(',')
else:
    print("args.render true, setting subjects_test to [args.viz_subject] which is ", [args.viz_subject])
    subjects_test = [args.viz_subject] #which is ['../vids/output.mp4']


semi_supervised = len(subjects_semi) > 0

if semi_supervised and not dataset.supports_semi_supervised():
    raise RuntimeError('Semi-supervised training is not implemented for this dataset')
            


def fetch(subjects, action_filter = None, subset = 1, parse_3d_poses = True):
    #initialize everything to empty arrays
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []


    for subject in subjects: #only will be one subject which is something like '../vids/output.mp4'
        #keypoints['../vids/output.mp4'].keys() should only contain 'custom'
        for action in keypoints[subject].keys():  #for just 'custom'                

            #sets poses_2d to the actual 3D array of keypoints (ig 4D since we put the 3D array into []??)
            poses_2d = keypoints[subject][action]

            #len(poses_2d) should just be 1
            print("LENGTH: ", len(poses_2d))

            for i in range(len(poses_2d)): # Iterate across cameras
                #append the 3D keypoints array to out_poses_2d
                out_poses_2d.append(poses_2d[i])

            
            #ALWAYS TRUE FOR NOW
            if subject in dataset.cameras():
                print("subject in dataset.cameras()")
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'

                #add the camera params to out_camera_params
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])
    
    #nullify out_camera_params if still empty
    if len(out_camera_params) == 0:
        out_camera_params = None

    #nullify out_poses_3d if still empty (WHICH ALWAYS IS)
    if len(out_poses_3d) == 0:
        out_poses_3d = None
    

    print("DOWNSAMPLE: ", args.downsample)
    stride = args.downsample #ALWAYS 1
    

    #return the camera parameters, 3d poses (None), and 2d poses (from Posenet)
    return out_camera_params, out_poses_3d, out_poses_2d



action_filter = None if args.actions == '*' else args.actions.split(',')


if action_filter is not None: #it's always coming up as None
    print('Selected actions:', action_filter)
    

#Should get camera params, None, and an array of 3D arrays of 2D keypoints, respectively
cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)

print("ARCHITECTURE: ", args.architecture)
filter_widths = [int(x) for x in args.architecture.split(',')] #this is [3, 3, 3, 3, 3], the filter dimension for convolution
print("FILTER_WIDTHS: ", filter_widths)

#ALWAYS TRUE
if not args.disable_optimizations and not args.dense and args.stride == 1:
    print("No disable opts or dense specified")


    #Use optimized model for single-frame predictions (this model is optimized for single-frame batching)
    #Pass initialization parameters 17, 2, 17, ...
    model_pos_train = TemporalModelOptimized1f(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                                filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels)

    
#instantiate a TemporalModel, the default model which uses temporal convolutions and can be used for all use cases
model_pos = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                            filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                            dense=args.dense)


'''When dealing with high-dimensional inputs such as images, it is impractical to connect neurons to 
all neurons in the previous volume. Instead, we connect each neuron to only a local region of the input
 volume. The spatial extent of this connectivity is a hyperparameter called the receptive 
 field of the neuron (equivalently this is the filter size).'''
receptive_field = model_pos.receptive_field()
print('INFO: Receptive field: {} frames'.format(receptive_field))

#Floor division // rounds result down to nearest whole number
pad = (receptive_field - 1) // 2 #Padding on each side

#Seems like causal convolutions are only for real-time. This makes it so that model’s output doesn’t depend on future inputs
if args.causal:
    print('INFO: Using causal convolutions')
    causal_shift = pad
else:
    print("INFO: NOT using causal convolutions")
    causal_shift = 0

model_params = 0

for parameter in model_pos.parameters():
    model_params += parameter.numel()

#Will have a LOT of trainable parameters
print('INFO: Trainable parameter count:', model_params)

#Set up cuda if it's available
if torch.cuda.is_available():
    model_pos = model_pos.cuda()
    model_pos_train = model_pos_train.cuda()
else:
    print("No cuda available")
    

#ALWAYS TRUE
if args.resume or args.evaluate:
    print("Resume or evaluate true")

    #Get and load the pretrained_h36m_detectron_coco.bin file
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)

    print('Loading checkpoint', chk_filename)

    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)

    print('This model was trained for {} epochs'.format(checkpoint['epoch']))

    #Load state dicts for both models
    model_pos_train.load_state_dict(checkpoint['model_pos'])
    model_pos.load_state_dict(checkpoint['model_pos'])
    

    model_traj=None
        
    
#Instantiate an UnchunkedGenerator. Non-batched data generator, used for testing. Sequences are returned one at a time
#(i.e. batch size = 1), without chunking. Pass the 2D Posenet poses, specifying args:
''' cameras -- list of cameras, one element for each video (optional, used for semi-supervised training) IGNORE
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training) IGNORE
    poses_2d -- list of input 2D keypoints, one element for each video
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled'''
test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
                                    pad=pad, causal_shift=causal_shift, augment=False,
                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
                    #THIS GENERATOR ONLY USED FOR TRAINING

#Log how many frames we're running the model on
print('INFO: Testing on {} frames'.format(test_generator.num_frames()))


#Evaluate
def evaluate(test_generator, action=None, return_predictions=False, use_trajectory_model=False):
    print("evaluate() called!")
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    with torch.no_grad():
        if not use_trajectory_model:
            model_pos.eval()
        else:
            model_traj.eval()
        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            # Positional model
            if not use_trajectory_model:
                predicted_3d_pos = model_pos(inputs_2d)
            else:
                predicted_3d_pos = model_traj(inputs_2d)

            # Test-time augmentation (if enabled)
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                if not use_trajectory_model:
                    predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)
                
            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()
                
            inputs_3d = torch.from_numpy(batch.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
            inputs_3d[:, :, 0] = 0    
            if test_generator.augment_enabled():
                inputs_3d = inputs_3d[:1]

            error = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()

            epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]
            
            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

            # Compute velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)
            
    if action is None:
        print('----------')
    else:
        print('----'+action+'----')
    e1 = (epoch_loss_3d_pos / N)*1000
    e2 = (epoch_loss_3d_pos_procrustes / N)*1000
    e3 = (epoch_loss_3d_pos_scale / N)*1000
    ev = (epoch_loss_3d_vel / N)*1000
    print('Test time augmentation:', test_generator.augment_enabled())
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
    print('Velocity Error (MPJVE):', ev, 'mm')
    print('----------')

    return e1, e2, e3, ev

#If we want to render a 3D visualization video
if args.render:
    print('Rendering...')
    
    #Get the input 2D keypoints again
    input_keypoints = keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()
    ground_truth = None

    #IGNORE
    if args.viz_subject in dataset.subjects() and args.viz_action in dataset[args.viz_subject]:
        if 'positions_3d' in dataset[args.viz_subject][args.viz_action]:
            ground_truth = dataset[args.viz_subject][args.viz_action]['positions_3d'][args.viz_camera].copy()
        else:
            print("NO positions_3d in dataset, skipping ground truth init")

    #IGNORE
    if ground_truth is None:
        print('INFO: this action is unlabeled. Ground truth will not be rendered.')
        

    #Instantiate an UnchunkedGenerator. Non-batched data generator, used for testing. Sequences are returned one at a time
    #(i.e. batch size = 1), without chunking. Pass the 2D Posenet poses, specifying args:
    ''' cameras -- list of cameras, one element for each video (optional, used for semi-supervised training) IGNORE
        poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training) IGNORE
        poses_2d -- list of input 2D keypoints, one element for each video
        pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
        causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
        augment -- augment the dataset by flipping poses horizontally
        kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
        joints_left and joints_right -- list of left/right 3D joints if flipping is enabled'''
    gen = UnchunkedGenerator(None, None, [input_keypoints],
                             pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)

    #Run evaluate on the generator for all frames
    prediction = evaluate(gen, return_predictions=True)

    #IGNORE
    if model_traj is not None and ground_truth is None:
        print("KEYED")
        prediction_traj = evaluate(gen, return_predictions=True, use_trajectory_model=True)
        prediction += prediction_traj
    
    #Can specify location for exporting the 3D joing positions, right now not doing this
    if args.viz_export is not None:
        print('Exporting joint positions to', args.viz_export)

        #Predictions are in camera space
        np.save(args.viz_export, prediction)
    

    #If there's an output video location specified, that means we need to render the animation video
    if args.viz_output is not None:
        #Invert camera transformation
        cam = dataset.cameras()[args.viz_subject][args.viz_camera]
        if ground_truth is not None:
            prediction = camera_to_world(prediction, R=cam['orientation'], t=cam['translation'])
            ground_truth = camera_to_world(ground_truth, R=cam['orientation'], t=cam['translation'])
        else:
            # If the ground truth is not available, take the camera extrinsic params from a random subject.
            # They are almost the same, and anyway, we only need this for visualization purposes.
            for subject in dataset.cameras():
                if 'orientation' in dataset.cameras()[subject][args.viz_camera]:
                    rot = dataset.cameras()[subject][args.viz_camera]['orientation']
                    break
            prediction = camera_to_world(prediction, R=rot, t=0)
            # We don't have the trajectory, but at least we can rebase the height
            prediction[:, :, 2] -= np.min(prediction[:, :, 2])
        
        anim_output = {'Reconstruction': prediction}
        if ground_truth is not None and not args.viz_no_ground_truth:
            anim_output['Ground truth'] = ground_truth
        
        input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])
        
        from common.visualization import render_animation
        render_animation(input_keypoints, keypoints_metadata, anim_output,
                         dataset.skeleton(), dataset.fps(), args.viz_bitrate, cam['azimuth'], args.viz_output,
                         limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                         input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
                         input_video_skip=args.viz_skip)
    

#If not rendering
else:
    print('Evaluating...')
    all_actions = {}
    all_actions_by_subject = {}
    for subject in subjects_test:
        if subject not in all_actions_by_subject:
            all_actions_by_subject[subject] = {}

        for action in dataset[subject].keys():
            action_name = action.split(' ')[0]
            if action_name not in all_actions:
                all_actions[action_name] = []
            if action_name not in all_actions_by_subject[subject]:
                all_actions_by_subject[subject][action_name] = []
            all_actions[action_name].append((subject, action))
            all_actions_by_subject[subject][action_name].append((subject, action))

    def fetch_actions(actions):
        out_poses_3d = []
        out_poses_2d = []

        for subject, action in actions:
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            poses_3d = dataset[subject][action]['positions_3d']
            assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
            for i in range(len(poses_3d)): # Iterate across cameras
                out_poses_3d.append(poses_3d[i])

        stride = args.downsample
        if stride > 1:
            # Downsample as requested
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]
        
        return out_poses_3d, out_poses_2d

    def run_evaluation(actions, action_filter=None):
        errors_p1 = []
        errors_p2 = []
        errors_p3 = []
        errors_vel = []

        for action_key in actions.keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action_key.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_act, poses_2d_act = fetch_actions(actions[action_key])
            gen = UnchunkedGenerator(None, poses_act, poses_2d_act,
                                     pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                     kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
            e1, e2, e3, ev = evaluate(gen, action_key)
            errors_p1.append(e1)
            errors_p2.append(e2)
            errors_p3.append(e3)
            errors_vel.append(ev)

        print('Protocol #1   (MPJPE) action-wise average:', round(np.mean(errors_p1), 1), 'mm')
        print('Protocol #2 (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 1), 'mm')
        print('Protocol #3 (N-MPJPE) action-wise average:', round(np.mean(errors_p3), 1), 'mm')
        print('Velocity      (MPJVE) action-wise average:', round(np.mean(errors_vel), 2), 'mm')

    if not args.by_subject:
        run_evaluation(all_actions, action_filter)
    else:
        for subject in all_actions_by_subject.keys():
            print('Evaluating on subject', subject)
            run_evaluation(all_actions_by_subject[subject], action_filter)
            print('')