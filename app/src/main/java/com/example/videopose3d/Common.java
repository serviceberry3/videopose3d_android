package com.example.videopose3d;

public class Common {

    public static int[][] keypoints_symmetry = {{1, 3, 5, 7, 9, 11, 13, 15}, {2, 4, 6, 8, 10, 12, 14, 16}};

    public static double[][] rot = {{0.14070565, -0.15007018, -0.7552408, 0.62232804}};

    public static int[][] skeleton_parents = {{-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15}};

    public static int[][] pairs = {{1,2}, {5,4},{6,5},{8,7},{8,9},{10,1},{11,10},{12,11},{13,1},{14,13},{15,14},{16,2},{16,3},{16,4},{16,7}};

    public static int[] kps_left = keypoints_symmetry[0];

    public static int[] kps_right = keypoints_symmetry[1];

    public static int[] joints_left = {4, 5, 6, 11, 12, 13};

    public static int[] joints_right = {1, 2, 3, 14, 15, 16};

    //padding on each side
    public static int pad = Math.floorDiv(243-1, 2);

    public static int causal_shift = 0;

    public static int [][] joint_pairs = {{0, 1}, {1, 3}, {0, 2}, {2, 4},
            {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},
            {5, 11}, {6, 12}, {11, 12},
            {11, 13}, {12, 14}, {13, 15}, {14, 16}};
}
