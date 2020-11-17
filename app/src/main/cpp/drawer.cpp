#include "drawer.h"

/**
     * Computes the length of a vector
     *
     * @param x x coordinate of a vector
     * @param y y coordinate of a vector
     * @param z z coordinate of a vector
     * @return the length of a vector
     */
float length(float x, float y, float z) {
    return (float) sqrt(x * x + y * y + z * z);
}

/**
     * Translates matrix m by x, y, and z in place.
     * @param m matrix
     * @param mOffset index into m where the matrix starts
     * @param x translation factor x
     * @param y translation factor y
     * @param z translation factor z
     */
void translateM(
        float* m, int mOffset,
        float x, float y, float z) {
    for (int i=0 ; i<4 ; i++) {
        int mi = mOffset + i;
        m[12 + mi] += m[mi] * x + m[4 + mi] * y + m[8 + mi] * z;
    }
}


/**
    * Define a viewing transformation in terms of an eye point, a center of
    * view, and an up vector.
    *
    * @param rm returns the result
    * @param rmOffset index into rm where the result matrix starts
    * @param eyeX eye point X
    * @param eyeY eye point Y
    * @param eyeZ eye point Z
    * @param centerX center of view X
    * @param centerY center of view Y
    * @param centerZ center of view Z
    * @param upX up vector X
    * @param upY up vector Y
    * @param upZ up vector Z
    */
void setLookAtM(float* rm, int rmOffset,
                float eyeX, float eyeY, float eyeZ,
                float centerX, float centerY, float centerZ, float upX, float upY,
                float upZ) {
    // See the OpenGL GLUT documentation for gluLookAt for a description
    // of the algorithm. We implement it in a straightforward way:
    float fx = centerX - eyeX;
    float fy = centerY - eyeY;
    float fz = centerZ - eyeZ;


    // Normalize f
    float rlf = 1.0f / length(fx, fy, fz);
    fx *= rlf;
    fy *= rlf;
    fz *= rlf;


    // compute s = f x up (x means "cross product")
    float sx = fy * upZ - fz * upY;
    float sy = fz * upX - fx * upZ;
    float sz = fx * upY - fy * upX;


    // and normalize s
    float rls = 1.0f / length(sx, sy, sz);
    sx *= rls;
    sy *= rls;
    sz *= rls;


    // compute u = s x f
    float ux = sy * fz - sz * fy;
    float uy = sz * fx - sx * fz;
    float uz = sx * fy - sy * fx;


    rm[rmOffset + 0] = sx;
    rm[rmOffset + 1] = ux;
    rm[rmOffset + 2] = -fx;
    rm[rmOffset + 3] = 0.0f;
    rm[rmOffset + 4] = sy;
    rm[rmOffset + 5] = uy;
    rm[rmOffset + 6] = -fy;
    rm[rmOffset + 7] = 0.0f;
    rm[rmOffset + 8] = sz;
    rm[rmOffset + 9] = uz;
    rm[rmOffset + 10] = -fz;
    rm[rmOffset + 11] = 0.0f;
    rm[rmOffset + 12] = 0.0f;
    rm[rmOffset + 13] = 0.0f;
    rm[rmOffset + 14] = 0.0f;
    rm[rmOffset + 15] = 1.0f;


    translateM(rm, rmOffset, -eyeX, -eyeY, -eyeZ);
}



void drawGrids() {
    //clear out the OpenGL buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //set the matrix mode
    glMatrixMode(GL_MODELVIEW);

    //make sure we're starting out with the identity matrix
    glLoadIdentity();

    //clear out what was there previously
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    //glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // white
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f); // black

    //enable GL drawing capabilities
    glEnable(GL_COLOR_MATERIAL);

    //the vertex array is enabled for writing and now used during rendering when glDrawArrays, or glDrawElements is called.
    glEnableClientState(GL_VERTEX_ARRAY);



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
    const float z1 = w * 0.60f;

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


    //glMultMatrixf(lookAtMat.ptr<GLfloat>(0));
    //glTranslatef(-viewPos.at<float>(0),-viewPos.at<float>(1),-viewPos.at<float>(2));

    //gluLookAt(viewPos.at<float>(0),viewPos.at<float>(1),viewPos.at<float>(2),twc.at<float>(0),twc.at<float>(1),twc.at<float>(2),0.0,1.0,0.0);

    //glPushMatrix pushes current matrix stack down by one, duplicating current matrix. After a glPushMatrix call, the matrix on top
    //of the stack is identical to the one below it.

    //(Initially, each of the stacks contains one matrix, an identity matrix.)

    //START OPERATING ON NEW MATRIX

    //gl MultMatrixf multiplies current matrix with the one specified (param), and replaces the current matrix with the product
    //Current matrix is determined by current matrix mode (see glMatrixMode). It is either projection matrix, modelview matrix,
    //or the texture matrix.

    //set line width of the camera drawing
    glLineWidth(1);

    //set the color of the camera drawing to white
    glColor4f(1.0f, 1.0f, 1.0f, 0.0f);


    GLfloat projectionMatrix[16];

    //create new perspective projection matrix. The height will stay the same while width will vary by aspect ratio.
    float ratio = 1.0f;
    float left = -ratio;
    float right = ratio;
    float bottom = -1.0f;
    float top = 1.0f;
    float near = 3.0f; //could try 1.0f?
    float far = 7.0f;  //could try 10.0f?


    //this projection matrix is applied to object coordinates in onDrawFrame()
    glFrustumf(left, right, bottom, top, near, far);


    //vertices of the lines to be drawn
    GLfloat vertexArray[240];

    float x =-0.5f, y=-0.5f, z=0.0f;

    for (int i = 0; i<120; i+=6) {
        vertexArray[i] = x;
        vertexArray[i+1] = y;
        vertexArray[i+2] = z;

        vertexArray[i+3] = -x;
        vertexArray[i+4] = y;
        vertexArray[i+5] = z;

        y+=0.05f;
    }

    x = -0.5f;
    y = -0.5f;
    z = 0.0f;

    for (int i = 120; i<240; i+=6) {
        vertexArray[i] = x;
        vertexArray[i+1] = y;
        vertexArray[i+2] = z;

        vertexArray[i+3] = x;
        vertexArray[i+4] = -y;
        vertexArray[i+5] = z;

        x+=0.05f;
    }

    //testing a triangle
    GLfloat triangleArray[] = {-0.5f, -0.25f, 0,
                               0.5f, -0.25f, 0,
                               0.0f,  0.559016994f, 0};

    GLfloat triangleCoords[] = {   // in counterclockwise order:
            0.0f, 0.622008459f, 0.0f, // top
            -0.5f, -0.311004243f, 0.0f, // bottom left
            0.5f, -0.311004243f, 0.0f  // bottom right
    };

    //specifies location and data format of an array of vertex coordinates to use when rendering.
    glVertexPointer(3, GL_FLOAT, 0, vertexArray); //2 coords/vertex, all floats, 0 byte stride from one vertex to next, loc at vertexArray

    //draw the camera on the screen
    //glDrawArrays(GL_LINES, 0, 80); //WAS 80

    //set up the camera

    //Position the eye (camera) in front of the origin
    float eyeX = 0.0f;
    float eyeY = 0.0f;
    float eyeZ = 3.3f;

    //We are looking forward toward the distance (position that the camera is looking at)
    float lookX = 0.0f;
    float lookY = 0.0f;
    float lookZ = 1.0f;

    //Set our up vector. This is where our head would be pointing were we holding the camera. The "up direction" of the camera
    float upX = 0.0f;
    float upY = 1.0f;
    float upZ = 0.0f;
    //For now set to straight up

    GLfloat viewMatrix[16];

    setLookAtM(viewMatrix, 0, eyeX, eyeY, eyeZ, lookX, lookY, lookZ, upX, upY, upZ);

    glMultMatrixf(viewMatrix);

    glDrawArrays(GL_LINES, 0, 80);

    glPushMatrix();

    //rotate grid
    glRotatef(90, 0, 1, 0);
    glTranslatef(-0.5, 0.0, -0.5);


    glDrawArrays(GL_LINES, 0, 80);

    glPopMatrix();

    glRotatef(90, 1, 0, 0);
    glTranslatef(0.0, 0.5, 0.5);

    glDrawArrays(GL_LINES, 0, 80);


    //glPopMatrix pops the current matrix stack, removing current mat and replacing it with the one below it on the stack.

    //Different GL implementations buffer commands in several different locations, including network buffers and graphics accelerator itself.
    //glFlush empties all of these buffers, causing all issued commands to be executed as quickly as they are accepted by actual rendering
    //engine. Though this execution may not be completed in any particular time period, it does complete in finite time.
    //Because any GL program might be executed over a network, or on an accelerator that buffers commands, all programs should call
    //glFlush whenever they count on having all of their previously issued commands completed. For example, call before waiting for user
    //input that depends on the generated image.
    glFlush();

    //Disable things
    //glDisableClientState(GL_VERTEX_ARRAY);
    //glDisable(GL_COLOR_MATERIAL);

}

void drawPerson() {

}






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
        //draw the background grids
        drawGrids();
    }
}






