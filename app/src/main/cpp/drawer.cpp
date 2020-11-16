#include "drawer.h"



//java interface functions
extern "C" {
    JNIEXPORT void JNICALL Java_com_example_jni_NdkHelper_glesInit(JNIEnv* env, jclass clazz) {
        glShadeModel(GL_SMOOTH);

        //clear out the color (this is when it gets cleaned to white during init)
        glClearColor(1.0f, 1.0f, 1.0f, 0.0f);

        //specifies depth val used by glClear to clear depth buffer. Values specified by glClearDepth are clamped to the range [0,1].
        glClearDepthf(1.0f);

        glEnable(GL_DEPTH_TEST);

        glDepthFunc(GL_LEQUAL);

        //perspective correction. Trivial
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
    }

    JNIEXPORT void JNICALL Java_com_example_jni_NdkHelper_glesRender(JNIEnv* env, jclass clazz) {

    }


    /*
    JNIEXPORT void Java_weiner_noah_ctojavaconnector_CircBuffer_circular_1buffer(JNIEnv *javaEnvironment, jclass obj, jlong sz, jint axis) {
        (axis==0 ? x_buff : y_buff) = new circular_buffer((size_t) sz);
    }

    JNIEXPORT void Java_weiner_noah_ctojavaconnector_CircBuffer_circular_1buffer_1destroy(JNIEnv *javaEnvironment, jclass obj, jint axis) {
        (axis==0 ? delete x_buff : delete y_buff);
    }

    JNIEXPORT void Java_weiner_noah_ctojavaconnector_CircBuffer_circular_1buf_1reset(JNIEnv* __unused javaEnvironment, jclass __unused obj, jint axis) {
        (axis==0 ? x_buff : y_buff)->circular_buf_reset();
    }

    JNIEXPORT jint Java_weiner_noah_ctojavaconnector_CircBuffer_circular_1buf_1put(JNIEnv* __unused javaEnvironment, jclass __unused obj, jfloat data, jint axis) {
    return (axis==0 ? x_buff : y_buff)->circular_buf_put(data);
    }*/
}






