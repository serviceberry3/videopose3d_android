package com.example.videopose3d;

/*

Non-batched data generator, used for testing.
    Sequences are returned one at a time (i.e. batch size = 1), without chunking.

    If data augmentation is enabled, the batches contain two sequences (i.e. batch size = 2),
    the second of which is a mirrored version of the first.


poses_2d -- list of input 2D keypoints, one element for each video
        pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
        causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
        augment -- augment the dataset by flipping poses horizontally
        kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
        joints_left and joints_right -- list of left/right 3D joints if flipping is enabled*/

public class UnchunkedGenerator {
    public boolean augment;
    public int[] kps_left;
    public int[] kps_right;
    public int[] jts_left;
    public int[] jts_right;
    public int pad, causal_shift;
    public float[][][] poses_2d;


    public UnchunkedGenerator(float[][][] kpts, int pad, int causal_shift, boolean augment, int[] kps_left, int[] kps_right, int[] jts_left, int[] jts_right) {
        this.augment = augment;
        this.kps_left = kps_left;
        this.kps_right = kps_right;
        this.jts_left = jts_left;
        this.jts_right = jts_right;
        this.pad = pad;
        this.causal_shift = causal_shift;
        poses_2d = kpts;
    }


    public int num_frames() {
        return 0;
    }

    public float[][][] pad_edges(float[][][] input, int padOnFront, int padOnBack) {
        float[][] front = input[0];
        float[][] back = input[input.length - 1];

        float[][][] ret = new float[input.length + padOnFront + padOnBack][][];

        for (int i = 0; i < ret.length; i++) {
            if (i < padOnFront) {
                ret[i] = front;
            }
            else if (i < padOnFront + input.length) {
                ret[i] = input[i - padOnFront];
            }
            else {
                ret[i] = back;
            }
        }

        return ret;
    }

    public float[][][][] concat4d(float[][][][] a1, float[][][][] a2, int axis) {
        float[][][][] ret = new float[2][][][];

        ret[0] = a1[0];
        ret[1] = a2[0];

        /*
        for (int i = 0; i < a1[0].length; i++) {
            ret[i] = a1[0][i];
        }

        for (int i = a1[0].length; i < ret.length; i++) {
            ret[i] = a2[0][i];
        }*/

        return ret;
    }

    public int[] concat1d(int[] a1, int[] a2, int axis) {
        int[] ret = new int[a1.length + a2.length];

        for (int i = 0; i < ret.length; i++) {
            if (i < a1.length)
                ret[i] = a1[i];
            else
                ret[i] = a2[i - a1.length];
        }

        return ret;
    }


    public float[][][][] next_epoch() {
        float[][][] padded2dKpts = pad_edges(poses_2d, 121, 121);

        float[][][][] newPadded2dKpts = new float[1][][][];

        newPadded2dKpts[0] = padded2dKpts;



        if (augment) {
            //append flipped version
            newPadded2dKpts = concat4d(newPadded2dKpts, newPadded2dKpts, 0);
            //newPadded2dKpts now contains 2 identical 3d arrays that are 272x17x2

            //flip all keypoints in second one
            for (int i = 0; i < newPadded2dKpts[1].length; i++) { //for 272
                for (int j = 0; j < newPadded2dKpts[1][i].length; j++) { //for 17
                    newPadded2dKpts[1][i][j][0] *= -1; //x coord
                    newPadded2dKpts[1][i][j][1] *= -1; //y coord
                }
            }

            int[] leftThenRt = concat1d(kps_left, kps_right, 0);
            int[] rtThenLeft = concat1d(kps_right, kps_left, 0);

            //do a lil switcheroo of the joints

            for (int i = 0; i < newPadded2dKpts[1].length; i++) { //for 272
                for (int j = 0; j < leftThenRt.length; j++) {
                    newPadded2dKpts[1][i][leftThenRt[j]] = newPadded2dKpts[1][i][rtThenLeft[j]];
                }
            }

        }


        return newPadded2dKpts;
    }
}
