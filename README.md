There's some demand for an Android app that implements Facebook Research's recent VideoPose3D model, and I'd like to use it for quadcopter computer vision purposes. I was originally going to plug the TfLite Posenet 2D human pose estimation model into a SLAM app and try to 3D map a human that way, but that's incovenient and probably much less effective than using the VideoPose3D model on Android.  

Note: using Detectron2 on Android would be too slow, so I'll try using TfLite Posenet to get the 2D human keypoints and then feeding them into Facebook's 3D model.  

UPDATE(01/07/21): The inference runs too slowly for Android right now. I looked into running it on the GPU to speed things up, which you might be able to do using Xiaomi MACE (tutorial is [here](https://v-hramchenko.medium.com/run-your-pytorch-model-on-android-gpu-using-libmace-7e43f623d95c)), but that library is meant for devices that have OpenCL support, and my device doesn't. It would probably still run too slowly. PyTorch recently released support for running models using Android's NNAPI, but so far I haven't been able to get it to work. I'm looking into substituting VideoPose3D for a lighter, faster model.

UPDATE(11/13/20): I've loaded the model into the Android app as a Torch Script. Fun visualization to come. I think each pair of Conv1D and BatchNorm1D layers can be fused together as well to improve runtime and memory usage.

Instructions:
run trace_model_cpu_og.py, which will store the output .pt in parent directory of this repo.
Copy the .pt file into the assets folder of the android src/main/ dir.

UPDATE(11/12/20): I've swapped out Detectron for Posenet and got the visualization working on my computer. You need to download the Posenet lite model: 
https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite. Place it in /home/[user]/Downloads/  

From root directory of my repo, you can run  
python3 infer_video_posenet.py --output-dir data --image-ext mp4 (relative path to videos folder). Then just run  

python3 run_3d_vis.py -d custom -k myvideos -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject (relative path to video) --viz-action custom --viz-camera 0 --viz-video (relative path to video) --viz-output (relative path to desired output location) --viz-size 6  

to do post-processing and render the visualization into desired output location.  

I also have live 3D visualization working: as long as you have the VideoPose3D cpn-pt-243.bin file downloaded and placed into [path to root of this repo]/../checkpoint/, just (from the root) run: 

 python3 3d_vis_realtime.py  
 
 with a webcam connected to video0 (tested in Ubuntu) to get live 3D visualization using Posenet and VideoPose3D. Make sure you also have all of the required packages installed for that pyscript, like qtgraph, python-opengl, opencv, etc.
