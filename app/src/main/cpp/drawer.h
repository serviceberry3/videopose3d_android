//
// Created by nodog on 11/16/20.
//

#ifndef VIDEOPOSE3D_DRAWER_H
#define VIDEOPOSE3D_DRAWER_H

#include <jni.h>
#include <GLES/gl.h>
#include <android/asset_manager_jni.h>

JavaVM* jvm;

/*
 * Class:     orb_slam2_android_nativefunc_OrbNdkHelper
 * Method:    initSystemWithParameters
 * Signature: (Ljava/lang/String;Ljava/lang/String;)V
 */
#ifdef __cplusplus
extern "C" {
#endif
    JNIEXPORT void JNICALL Java_com_example_jni_NdkHelper_glesInit(JNIEnv* env, jclass clazz);

    JNIEXPORT void JNICALL Java_orb_slam2_android_nativefunc_OrbNdkHelper_glesRender(JNIEnv* env, jclass clazz);

#ifdef __cplusplus
}
#endif //__cplusplus
#endif //VIDEOPOSE3D_DRAWER_H