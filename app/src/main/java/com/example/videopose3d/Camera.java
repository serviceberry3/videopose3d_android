package com.example.videopose3d;

public class Camera {
    public static float[][][] normalize_screen_coordinates(float[][][] coords, int w, int h) {
        //Normalize so that [0, w] is mapped to [-1, 1], while preserving aspect ratio

        for (int i = 0; i < coords.length; i++) {
            for (int k = 0; k < coords[i].length; k++){
                for (int j = 0; j < coords[i][k].length; j++) {
                    coords[i][k][j] /= w * 2;
                    if (j == 0)
                        coords[i][k][j] -= 1;
                    else
                        coords[i][k][j] -= (float) h / w;
                }
            }
        }

        return coords;
    }

    public static float[][] normalize_screen_coordinates_new(float[][] coords, int w, int h) {
        for (int i = 0; i < coords.length; i++) {
            for (int j = 0; j < coords[i].length; j++) {
                if (j==0) {
                    coords[i][j] -= (float) w / 2;
                    coords[i][j] /= (float) w / 2;
                }

                else {
                    coords[i][j] -= (float) h / 2;
                    coords[i][j] /= (float) h / 2;
                }
            }
        }

        return coords;
    }

    public static float[][] camera_to_world(float[][] coords, double[][] rotation, int translation) {
        return null;
    }
}
