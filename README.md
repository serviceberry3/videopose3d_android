There's some demand for an Android app that implements Facebook Research's recent VideoPose3D model, and I'd like to use it for quadcopter computer vision purposes. I was originally going to plug the TfLite Posenet 2D human pose estimation model into a SLAM app and try to 3D map a human that way, but that's incovenient and probably much less effective than using the VideoPose3D model on Android.   

Possible problems:  

- Porting into Android/PyTorch - could get very complicated
- Speed: using Detectron2 on Android would be too slow, so I'll try using TfLite Posenet to get the 2D human keypoints and then feeding them into Facebook's 3D model
- Real time: configuring the app to read in frames from real time and do the inference might be tricky
