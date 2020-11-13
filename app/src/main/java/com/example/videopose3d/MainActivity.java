package com.example.videopose3d;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.hardware.SensorEventListener;
import android.location.LocationListener;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.util.Log;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import org.pytorch.IValue;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

public class MainActivity extends AppCompatActivity implements GLSurfaceView.Renderer, CameraBridgeViewBase.CvCameraViewListener2, View.OnClickListener {
    public final String TAG = "MainAct";

    ImageView imgDealed;

    LinearLayout linear;

    String vocPath, calibrationPath;

    private static final int INIT_FINISHED = 0x00010001;

    private CameraBridgeViewBase mOpenCvCameraView;
    private boolean mIsJavaCamera = true;
    private MenuItem mItemSwitchCamera = null;

    private final int CONTEXT_CLIENT_VERSION = 3;

    //OpenGL SurfaceView
    private GLSurfaceView mGLSurfaceView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);//will hide the title.
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getSupportActionBar().hide(); //hide the title bar.

        //set the view to main act view
        setContentView(R.layout.activity_main);



        imgDealed = (ImageView) findViewById(R.id.img_dealed);

        //mIsJavaCamera is bool describing whether or not we're using JavaCameraView. Which we always are, it seems.

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.java_cam_view);


        //check the the view was found by ID and all is well
        if (mOpenCvCameraView == null) {
            Log.e(TAG, "mOpenCvCameraView came up null");
        }
        else {
            Log.d(TAG, "mOpenCvCameraView non-null, OK");
        }

        //make our OpenCvCameraView visible and set the listener for the camera
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        //instantiate new GLSurfaceView
        mGLSurfaceView = new GLSurfaceView(this);
        linear = (LinearLayout) findViewById(R.id.surfaceLinear);

        //mGLSurfaceView.setEGLContextClientVersion(CONTEXT_CLIENT_VERSION);

        //set the renderer for the GLSurfaceView
        mGLSurfaceView.setRenderer(this);

        linear.addView(mGLSurfaceView, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.MATCH_PARENT));


        Bitmap bitmap = null;
        Module module = null;

        try {
            //Load model: loading serialized torchscript module from packaged into app android asset model.pt,
            module = Module.load(assetFilePath(this, "processed_mod.pt"));
        }

        catch (IOException e) {
            Log.e("PytorchHelloWorld", "Error reading assets", e);
            finish();
        }

        Log.i("DBUG", "Read in VideoPose3D successfully");

        /*
        //Preparing input tensor from the image (in torchvision format)
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
        //inputTensor’s shape is 1x3xHxW, where H and W are bitmap height and width appropriately.


        //Running the model - run loaded module’s forward method, get result as org.pytorch.Tensor outputTensor with shape 1x1000
        assert module != null;
        final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();


        //Get output tensor content as java array of floats
        final float[] scores = outputTensor.getDataAsFloatArray(); //returns java array of floats with scores for every image net class
         */
    }

    //use this OpenCV loader callback to instantiate Mat objects, otherwise we'll get an error about Mat not being found
    public BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            Log.i(TAG, "BaseLoaderCallback called!");

            if (status == LoaderCallbackInterface.SUCCESS) {//instantiate everything we need from OpenCV
                //everything succeeded
                Log.i(TAG, "OpenCV loaded successfully, everything created");
                mOpenCvCameraView.enableView();
            }

            else {
                super.onManagerConnected(status);
            }
        }
    };

    @Override
    protected void onStart() {
        super.onStart();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        }

        else {
            Log.d("OpenCV", "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        mGLSurfaceView.onPause();

        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();

    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }


    /**
     * Copies specified asset to the file in /files app directory and returns this file absolute path.
     *
     * @return absolute file path
     */
    public static String assetFilePath(Context context, String assetName) throws IOException {
        //Looking for file [files directory]/assetname - here files directory will be "assets"
        File file = new File(context.getFilesDir(), assetName);
        Log.i("DBUG", file.getAbsolutePath());

        //Default: return absolute path of file
        if (file.exists() && file.length() > 0) {
            Log.i("DBUG", "Found specified file, returning abs path");
            return file.getAbsolutePath();
        }


        Log.i("DBUG", "Specified file doesn't exist or was empty");
        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    @Override
    public void onSurfaceCreated(GL10 gl10, EGLConfig eglConfig) {

    }

    @Override
    public void onSurfaceChanged(GL10 gl10, int i, int i1) {

    }

    @Override
    public void onDrawFrame(GL10 gl10) {

    }

    @Override
    public void onClick(View view) {

    }

    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        //get the frame as a mat in RGBA format
        Mat im = inputFrame.rgba();

        //whatever gets returned here is what's displayed
        return im;
    }
}