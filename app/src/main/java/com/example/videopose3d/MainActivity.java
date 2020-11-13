package com.example.videopose3d;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.Window;
import android.widget.ImageView;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import org.pytorch.IValue;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {
    public final String TAG = "MainAct";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);//will hide the title.
        getSupportActionBar().hide(); //hide the title bar.


        setContentView(R.layout.activity_main);



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
            }

            else {
                super.onManagerConnected(status);
            }
        }
    };

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this,
                    mLoaderCallback);
        }

        else {
            Log.d("OpenCV", "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
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
}