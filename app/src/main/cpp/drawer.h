#ifndef VIDEOPOSE3D_DRAWER_H
#define VIDEOPOSE3D_DRAWER_H

#include <jni.h>
#include <GLES/gl.h>
#include <GLES/glext.h>
#include <GLES/glplatform.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <cmath>
//#include "/home/nodog/Downloads/glues-1.3/glues/source/glues.h"

#define LOG_TAG "ORB_SLAM_SYSTEM_MAPDRAWER"

#define LOG(...) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG, __VA_ARGS__)


JavaVM* jvm;

#ifdef __cplusplus
extern "C" {
#endif
    JNIEXPORT void JNICALL Java_com_example_jni_NdkHelper_glesInit(JNIEnv* env, jclass clazz);

    JNIEXPORT void JNICALL Java_orb_slam2_android_nativefunc_OrbNdkHelper_glesRender(JNIEnv* env, jclass clazz);

#ifdef __cplusplus
}
#endif //__cplusplus
#endif //VIDEOPOSE3D_DRAWER_H