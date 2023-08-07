package com.example.tflite_yolov7_test;

public class TfliteRunMode {
    public enum Mode{
        NONE_FP32,
        NONE_FP16,
        NONE_INT8,
        GPU_FP32,
        GPU_FP16,
        GPU_INT8,
        NNAPI_GPU_FP32,
        NNAPI_GPU_FP16,
        NNAPI_DSP_INT8
    }
    static public boolean isQuantizedMode(Mode mode){
        return mode == Mode.NONE_INT8 || mode == Mode.GPU_INT8 || mode == Mode.NNAPI_DSP_INT8;
    }
}
