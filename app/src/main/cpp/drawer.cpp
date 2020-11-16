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
        //clear out the OpenGL buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //set the matrix mode
        glMatrixMode(GL_MODELVIEW);

        //make sure we're starting out with the identity matrix
        glLoadIdentity();

        //clear out what was there previously
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);



        LOG("Clearing color to black...");
        //glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // white
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f); // black

        //generate 4x4 identity matrix
        cv::Mat temp = cv::Mat::eye(4, 4, CV_32F);

        //enable GL drawing capabilities
        glEnable (GL_COLOR_MATERIAL);

        LOG("Enabling vertex arrays...");
        //the vertex array is enabled for writing and now used during rendering when glDrawArrays, or glDrawElements is called.
        glEnableClientState (GL_VERTEX_ARRAY);


        LOG("Creating rotation and translation matrices");

        //instantiate a 3x3 camera-to-world rotation matrix
        cv::Mat Rwc(3,3,CV_32F);

        //instantiate a 3x1 camera-to-world translation matrix
        cv::Mat twc(3,1,CV_32F);

        //instantiate a 3x1 lookAt matrix
        cv::Mat viewPos(3,1,CV_32F);

        //lookAt matrix generation with these values
        viewPos.at<float>(0) = 0.0;
        viewPos.at<float>(1) = 0.0;
        viewPos.at<float>(2) = 0.3;

        //start lookAtMat it as a 4x4 identity matrix
        cv::Mat lookAtMat = cv::Mat::eye(4, 4, CV_32F);

        //a couple of 3x1 matrices (actually vectors)
        cv::Mat zVec(3, 1, CV_32F);
        cv::Mat sVec(3, 1, CV_32F);
        cv::Mat uVec(3, 1, CV_32F);


        const float &w = 0.08f;
        const float h = w * 0.75f;
        const float z = w * 0.60f;

        //rotation and translation mats

        cv::Mat mCameraPose(3, 4, CV_32F);

        mCameraPose.at<float>(0, 0) = 0.6;
        mCameraPose.at<float>(0, 1) = -0.8;
        mCameraPose.at<float>(0, 2) = 1;
        mCameraPose.at<float>(0, 3) = 1;
        mCameraPose.at<float>(1, 0) = 0.8;
        mCameraPose.at<float>(1, 1) = 0.6;
        mCameraPose.at<float>(1, 2) = 1;
        mCameraPose.at<float>(1, 3) = 1;
        mCameraPose.at<float>(2, 0) = 1;
        mCameraPose.at<float>(2, 1) = 1;
        mCameraPose.at<float>(2, 2) = 1;
        mCameraPose.at<float>(2, 3) = 1;


        //rotation matrix will be transpose of first 3x3 of Tcw
        Rwc = mCameraPose.rowRange(0,3).colRange(0,3).t();

        //use last 3x1 for translation matrix
        twc = -Rwc * mCameraPose.rowRange(0,3).col(3);

        //lookAt matrix
        zVec = -Rwc * viewPos;
        normalize(zVec.col(0),zVec.col(0),1,cv::NORM_L2);
        uVec.at<float>(0)=0.0;
        uVec.at<float>(1)=1.0;
        uVec.at<float>(2)=0.0;
        sVec = zVec.cross(uVec);
        normalize(sVec.col(0),sVec.col(0),1,cv::NORM_L2);
        uVec = sVec.cross(zVec);
        normalize(uVec.col(0),uVec.col(0),1,cv::NORM_L2);
        //uVec = sVec.cross(zVec);

        viewPos = Rwc * viewPos + twc;


        //fill the lookat matrix with preset values
        lookAtMat.at<float>(0,0) = sVec.at<float>(0);
        lookAtMat.at<float>(1,0) = sVec.at<float>(1);
        lookAtMat.at<float>(2,0) = sVec.at<float>(2);
        lookAtMat.at<float>(3,0) = 0.0;
        lookAtMat.at<float>(0,1) = uVec.at<float>(0);
        lookAtMat.at<float>(1,1) = uVec.at<float>(1);
        lookAtMat.at<float>(2,1) = uVec.at<float>(2);
        lookAtMat.at<float>(3,1) = 0.0;
        lookAtMat.at<float>(0,2) = -zVec.at<float>(0);
        lookAtMat.at<float>(1,2) = -zVec.at<float>(1);
        lookAtMat.at<float>(2,2) = -zVec.at<float>(2);
        lookAtMat.at<float>(3,2) = 0.0;
        lookAtMat.at<float>(0,3) = 0.0;
        lookAtMat.at<float>(1,3) = 0.0;
        lookAtMat.at<float>(2,3) = 0.0;
        lookAtMat.at<float>(3,3) = 1.0;

        /*glMultMatrixf(lookAtMat.ptr<GLfloat>(0));
        glTranslatef(-viewPos.at<float>(0),-viewPos.at<float>(1),-viewPos.at<float>(2));*/



        //gluLookAt(viewPos.at<float>(0),viewPos.at<float>(1),viewPos.at<float>(2),twc.at<float>(0),twc.at<float>(1),twc.at<float>(2),0.0,1.0,0.0);

        //glPushMatrix pushes current matrix stack down by one, duplicating current matrix. After a glPushMatrix call, the matrix on top
        //of the stack is identical to the one below it.

        //(Initially, each of the stacks contains one matrix, an identity matrix.)
        //glPushMatrix();

        //START OPERATING ON NEW MATRIX

        //multiplies current matrix with the one specified (param), and replaces the current matrix with the product
        //Current matrix is determined by current matrix mode (see glMatrixMode). It is either projection matrix, modelview matrix,
        //or the texture matrix.
        glMultMatrixf(temp.ptr<GLfloat>(0));

        //set line width of the camera drawing
        glLineWidth(3);

        //set the color of the camera drawing (now GREEN)
        glColor4f(0.0f, 1.0f, 0.0f, 1.0f);

        //vertices of the camera to be drawn
        GLfloat vertexArray[] = { 0, 0, 0, w, h, z, 0, 0, 0, w, -h, z, 0, 0, 0, -w,
                                  -h, z, 0, 0, 0, -w, h, z, w, h, z, w, -h, z, -w, h, z, -w, -h, z,
                                  -w, h, z, w, h, z, -w, -h, z, w, -h, z, };

        GLfloat triangleArray[] = {-0.5f, -0.25f, 0,
                                   0.5f, -0.25f, 0,
                                   0.0f,  0.559016994f, 0
        };

        //specifies location and data format of an array of vertex coordinates to use when rendering.
        glVertexPointer(3, GL_FLOAT, 0, vertexArray); //3 coords/vertex, all floats, 0 byte stride from one vertex to next, loc at vertexArray

        //draw the camera on the screen
        glDrawArrays(GL_LINES, 0, 16);

        //glPopMatrix pops the current matrix stack, removing current mat and replacing it with the one below it on the stack.
        //glPopMatrix();

        //Different GL implementations buffer commands in several different locations, including network buffers and graphics accelerator itself.
        //glFlush empties all of these buffers, causing all issued commands to be executed as quickly as they are accepted by actual rendering
        //engine. Though this execution may not be completed in any particular time period, it does complete in finite time.
        //Because any GL program might be executed over a network, or on an accelerator that buffers commands, all programs should call
        //glFlush whenever they count on having all of their previously issued commands completed. For example, call before waiting for user
        //input that depends on the generated image.
        glFlush();


        //glDisableClientState (GL_VERTEX_ARRAY);
        //glDisable (GL_COLOR_MATERIAL);
    }
}






