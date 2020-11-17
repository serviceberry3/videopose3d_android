package com.example.videopose3d;

public class Camera {
    public static float[][] normalize_screen_coordinates(float[][] coords, int w, int h) {
        //Normalize so that [0, w] is mapped to [-1, 1], while preserving aspect ratio

        for (int i = 0; i < coords.length; i++) {
            
        }
    }


/*
    def normalize_screen_coordinates(X, w, h):
            assert X.shape[-1] == 2

            # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X/w*2 - [1, h/w]

    def normalize_screen_coordinates_new(X, w, h):
            assert X.shape[-1] == 2

            return (X -(w/2, h/2) ) / (w/2, h/2)*/


    public static
}
