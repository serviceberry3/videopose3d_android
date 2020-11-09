import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import cv2 as cv
import math
import numpy
import sys


# The model output consist of 2 parts:
# - heatmaps (9,9,17) - corresponds to the probability of appearance of 
# each keypoint in the particular part of the image (9,9)(without applying sigmoid 
# function). Is used to locate the approximate position of the joint. (There are 17 heatmaps.)
# - offset vectors (9,9,34) is called offset vectors. Is used for more exact
#  calculation of the keypoint's position. First 17 of the third dimension correspond
# to the x coordinates and the second 17 of them correspond to the y coordinates

#With heatmaps we can find approximate positions of the joints. After findingindex for maximum value we
#upscale it w/output stride value and size of input tensor. After that we can adjust positions w/offset vectors.

#Pseudocode for parsing output:
#for every keypoint in heatmap_data:
#    1. find indices of max values in the 9x9 grid
#    2. calculate position of the keypoint in the image
#    3. adjust the position with offset_data
#    4. get the maximum probability
 
#    if max probability > threshold:
#      if the position lies inside the shape of resized image:
#        set the flag for visualisation to True
def parse_output(heatmap_data, offset_data, threshold):
    '''
    Input:
    heatmap_data - heatmaps for an image. 3D array
    offset_data - offset vectors for an image. 3D array
    threshold - probability threshold for the keypoints. Scalar value
    Output:
    array with coordinates of the keypoints and flags for those that have
    low probability
    '''

    #get number of joints
    joint_num = heatmap_data.shape[-1]
    #print("Num of joints is ", joint_num)

    #create 2D array of zeros (17 x 3) to hold joint coordinates and score for each joint
    pose_kps = np.zeros((joint_num, 3), np.uint32)

    '''
    print("HEATMAP_DATA")
    print(heatmap_data) #heatmap_data is an array of 9 9x17 arrays (the 9x9 heatmap for all 17 joints)'''

    #iterate for all joints
    for i in range(joint_num):
        #get 9x9 heatmap for this joint
        joint_heatmap = heatmap_data[..., i] #ellipsis may be used to indicate selecting in full any remaining unspecified dimensions.
        #equivalent to heatmap_data[:, :, i], which indexes all 9x17 arrays, all 9 rows, just this joint column

        '''
        print("JOINT_HEATMAP")
        print(joint_heatmap)

        print("ARGWHERE")'''

        #find maximum probability that this joint is found in any heatmap square
        max_prob = np.max(joint_heatmap)

        #print(np.argwhere(joint_heatmap==max_prob))

        #find the position of the max value in the heatmap
        max_val_pos = np.squeeze(np.argwhere(joint_heatmap == max_prob))
        #the argwhere finds indices of array element that's the np.max. It'll have shape (N, array.ndim) where N is number of
        #elements in array that match (1 in this case). So shape will be (1, 2) since joint_heatmap is 2d array (9x9)
        #squeeze converts the array to 1D

        #divide the max_val_pos array members by 8 (since row/col indices start at 0), mult by 257, and convert to int to get actual coordinates of keypoint
        remap_pos = np.array(max_val_pos / 8 * 257, dtype = np.int32)

        #get appropriate offset value and add it to the row coord of the keypoint, then store x coord in pose_kps
        pose_kps[i, 0] = int(remap_pos[0] + offset_data[max_val_pos[0], max_val_pos[1], i])

        #get appropriate offset value (need to index into the second 17 of offset vectors' third dim as noted above), add it to column coord, and store
        pose_kps[i, 1] = int(remap_pos[1] + offset_data[max_val_pos[0], max_val_pos[1], i + joint_num])


        #if we're confident enough that this joint was found
        if max_prob > threshold:
            #if we know that this keypoint was found INSIDE the image
            if pose_kps[i, 0] < 257 and pose_kps[i, 1] < 257:
                #add an adjusted score of "1" to third slot of this keypoint
                pose_kps[i, 2] = 1

    #return array of keypoints
    return pose_kps


def draw_kps(show_img, kps, ratio = None):
    #iterate over all keypoints (kps is 2D 17 x 3 array)
    for i in range(kps.shape[0]):
        #make sure the score of this kp equals 1
        if kps[i, 2]:
            #make sure ratio is of type TUPLE (make sure a ratio was passed)
            if isinstance(ratio, tuple):
                #draw a circle using column as x coord and row as y coord, passing scalar to determine color, radius 2, and the desired thickness
                cv.circle(show_img, (int(round(kps[i, 1] * ratio[1])), int(round(kps[i, 0] * ratio[0]))), 2, (0, 255, 255), round(int(1 * ratio[1])))

                #jump to next iteration of loop to skip next line
                continue

            #otherwise draw circle without ratio scaling
            cv.circle(show_img, (kps[i, 1], kps[i, 0]), 2, (0, 255, 255), 5)
    
    #return the keypointed image
    return show_img


def estimate_pose(image_src):
    #don't use ellipsis to truncate arrays when printing
    np.set_printoptions(threshold=sys.maxsize)

    #set path to model
    path = "/home/nodog/Downloads/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"

    #name of pop-up
    window_name = "Original image"

    #instructions
    #print("Press any key to verify image and continue")

    #get actual height and width of input image
    actual_height = image_src.shape[0]
    actual_width = image_src.shape[1]

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

    #calculate scaling ratios to scale up the keypoints later
    height_scaler = actual_height / height
    width_scaler = actual_width / width

    #print("Need height ", height, "and width ", width, "for model")

    #resize input image (testing right now on one image)
    image_new = cv.resize(image_src, (width, height))

    '''OPTIONAL

    #display the resized image in pop-up window
    cv.imshow(window_name, image_new)

    #waits for user to press any key  
    cv.waitKey(0)  
    
    #closing all open windows  
    cv.destroyAllWindows()'''

    #print(image_new)

    #add a new dimension to the image mat to match model's input requirements
    input = np.expand_dims(image_new.copy(), axis = 0)

    #check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    #Floating point models offer the best accuracy, at the expense of model size 
    #and performance. GPU acceleration requires use of floating point models.

    #Model requires incoming imgs to have float values. Here we normalize them to be in range from 0 to 1 by subtracting
    #default mean and dividing by default standard deviation.
    if floating_model:
        input = (np.float32(input) - 127.5) / 127.5

    #Process image
    #Sets the value of the input tensor
    interpreter.set_tensor(input_details[0]['index'], input)

    #Run the actual model inference
    interpreter.invoke()

    #Extract output data from the interpreter
    heatmap_data = interpreter.get_tensor(output_details[0]['index'])
    offset_data = interpreter.get_tensor(output_details[1]['index'])

    #Getting rid of the extra dimension we added
    heatmaps = np.squeeze(heatmap_data)
    offsets = np.squeeze(offset_data)

    #print("Heatmaps shape: ", heatmaps.shape)
    #print("Offsets shape: ", offsets.shape)

    '''
    #make copy of expanded input image with all vals multiplied by 127.5, increased by 127.5, then divided by 255 (undo normalization done above)
    show = np.squeeze((input.copy() * 127.5 + 127.5) / 255.0)

    #multiply all values in show by 255 and convert to ints
    show = np.array(show * 255, np.uint8)'''

    #get the keypoints, confidence threshold pretty generous
    kps = parse_output(heatmaps, offsets, 0.3)

    #cv.imshow("Original image", image_src.copy())

    #scale the keypoints back up
    #print(kps)

    for kp in kps:
        kp[0] *= height_scaler
        kp[1] *= width_scaler

    #print(kps)


    '''OPTIONAL
    #show image with keypoints drawn
    cv.imshow("Keyed image", draw_kps(image_src.copy(), kps))

    #instructions
    print("Press any key to finish")

    #waits for user to press any key  
    cv.waitKey(0)
    
    #closing all open windows  
    cv.destroyAllWindows()'''

    #return the keypoints
    return kps

    '''
    #quit program
    exit()'''







