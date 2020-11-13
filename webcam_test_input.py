import cv2 
  
  
#Define vid as capture feed from video0
vid = cv2.VideoCapture(0) 


import numpy as np

def rotateImage(image, angle):
    image0 = image

    if hasattr(image, 'shape'):

        image_center = tuple(np.array(image.shape)[:2]/2)
        shape = tuple(image.shape)


    elif hasattr(image, 'width') and hasattr(image, 'height'):

        image_center = tuple(np.array((image.width/2, image.height/2)))

        shape = (image.width, image.height)


    else:
        print("Can't get dims of input")


    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    image = np.asarray( image[:,:] )

    rotated_image = cv2.warpAffine(image, rot_mat, shape, flags = cv2.INTER_LINEAR)

    # Copy the rotated data back into the original image object.
    cv2.cv.SetData(image0, rotated_image.tostring())

    return image0
  

#Loop until quit key pressed
while (True): 
    #Capture the video frame by frame 
    ret, frame = vid.read()

    #Rotate the input
    #cv2.flip(frame, flipCode=-1)


    flipped = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    #cv2.imshow('frame',flipped)
  
    #Display the resulting frame 
    cv2.imshow('INPUT', flipped) 
      
    #'q' button is set as quitting button
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

#Release the cap object 
vid.release() 

#Destroy all open pop-up windows 
cv2.destroyAllWindows() 
