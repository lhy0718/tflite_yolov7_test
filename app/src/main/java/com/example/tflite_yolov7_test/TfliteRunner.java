package com.example.tflite_yolov7_test;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;

import org.tensorflow.lite.DelegateFactory;
import org.tensorflow.lite.InterpreterApi;
import org.tensorflow.lite.gpu.GpuDelegateFactory;
import org.tensorflow.lite.nnapi.NnApiDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import com.example.tflite_yolov7_test.TfliteRunMode.*;
import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.Tasks;
import com.google.android.gms.tflite.client.TfLiteInitializationOptions;
import com.google.android.gms.tflite.gpu.support.TfLiteGpu;
import com.google.android.gms.tflite.java.TfLite;

public class TfliteRunner {
    final int numBytesPerChannel_float = 4;
    final int numBytesPerChannel_int = 1;
    static {
        System.loadLibrary("native-lib");
    }
    private Integer[] outputOrder;
    public native float[][] postprocess(float[][][][] out1, float[][][][] out2, float[][][][] out3, int inputSize, float conf_thresh, float iou_thresh);
    public InterpreterApi tfliteInterpreter = null;
    Mode runmode;
    int inputSize;
    class InferenceRawResult{
        public float[][][][] out1;
        public float[][][][] out2;
        public float[][][][] out3;

        public InferenceRawResult(int inputSize){
            this.out1 = new float[1][inputSize/8][inputSize/8][3*85];
            this.out2 = new float[1][inputSize/16][inputSize/16][3*85];
            this.out3 = new float[1][inputSize/32][inputSize/32][3*85];
        }
    }
    Object[] inputArray;
    Map<Integer, Object> outputMap;
    InferenceRawResult rawres;
    float conf_thresh;
    float iou_thresh;
    public Task<Void> initializeTask;
    public TfliteRunner(Context context, String baseModelName, Mode runmode, int inputSize, float conf_thresh, float iou_thresh) throws Exception{
        this.runmode = runmode;
        this.rawres = new InferenceRawResult(inputSize);
        this.inputSize = inputSize;
        this.conf_thresh = conf_thresh;
        this.iou_thresh = iou_thresh;
        Log.i("TfliteRunner", "runmode: " + runmode.toString() + ", inputSize: " + inputSize);
        loadModel(context, baseModelName, runmode, inputSize, 4);
    }
    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    public void loadModel(Context context, String baseModelName, Mode runmode, int inputSize, int numThreads) throws Exception{
        String precisionStr;
        InterpreterApi.Options interpreterOption = new InterpreterApi.Options()
            .setNumThreads(numThreads)
            .setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY);

        this.initializeTask = TfLite.initialize(
            context,
            TfLiteInitializationOptions.builder()
                .setEnableGpuDelegateSupport(true)
                .build()
        );
        this.outputOrder = new Integer[]{1, 2, 0};

        switch (runmode){
            case NONE_FP32:
                precisionStr = "float32";
                interpreterOption.setUseXNNPACK(true);
                break;
            case NONE_FP16:
                precisionStr = "float16";
                interpreterOption.setUseXNNPACK(true);
                break;
            case NONE_INT8:
                precisionStr = "int8";
                interpreterOption.setUseXNNPACK(true);
                break;
            case GPU_FP32:
                precisionStr = "float32";
                interpreterOption.addDelegateFactory(new GpuDelegateFactory());
                break;
            case GPU_FP16:
                precisionStr = "float16";
                interpreterOption.addDelegateFactory(new GpuDelegateFactory());
                break;
            case GPU_INT8:
                precisionStr = "int8";
                interpreterOption.addDelegateFactory(new GpuDelegateFactory());
                break;
            case NNAPI_GPU_FP32:
                precisionStr = "float32";
                interpreterOption.addDelegateFactory((DelegateFactory) new NnApiDelegate());
                interpreterOption.setUseNNAPI(true);
                break;
            case NNAPI_GPU_FP16:
                precisionStr = "float16";
                interpreterOption.setUseNNAPI(true);
                break;
            case NNAPI_DSP_INT8:
                precisionStr = "int8";
                interpreterOption.setUseNNAPI(true);
                break;
            default:
                throw new RuntimeException("Unknown run mode!");
        }

        String modelName = baseModelName + "_" + precisionStr + "_" + inputSize + ".tflite";
        MappedByteBuffer tfliteModelBuf = TfliteRunner.loadModelFile(context.getAssets(), modelName);

        InterpreterApi.Options finalInterpreterOption = interpreterOption;
        this.initializeTask.addOnSuccessListener(a -> {
            this.tfliteInterpreter = InterpreterApi.create(
                tfliteModelBuf,
                finalInterpreterOption
            );
        })
        .addOnFailureListener(e -> Log.e("tfliteInterpreter", String.format("Cannot initialize interpreter: %s",e.getMessage())));
    }
    static public Bitmap getResizedImage(Bitmap bitmap, int inputSize) {
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true);
        return resized;
    }
    public void setInput(Bitmap resizedbitmap){
        ByteBuffer imgData = ByteBuffer.allocateDirect(inputSize * inputSize * 3 * 4);

        int[] intValues = new int[inputSize * inputSize];
        resizedbitmap.getPixels(intValues, 0, resizedbitmap.getWidth(), 0, 0, resizedbitmap.getWidth(), resizedbitmap.getHeight());

        imgData.order(ByteOrder.nativeOrder());
        imgData.rewind();
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                int pixelValue = intValues[i * inputSize + j];
                float r = (((pixelValue >> 16) & 0xFF)) / 255.0f;
                float g = (((pixelValue >> 8) & 0xFF)) / 255.0f;
                float b = ((pixelValue & 0xFF)) / 255.0f;
                imgData.putFloat(r);
                imgData.putFloat(g);
                imgData.putFloat(b);
            }
        }
        this.inputArray = new Object[]{imgData};
        this.outputMap = new HashMap<>();
        outputMap.put(this.outputOrder[0], this.rawres.out1); // float[1][inputSize/8][inputSize/8][3*85];
        outputMap.put(this.outputOrder[1], this.rawres.out2); // float[1][inputSize/16][inputSize/16][3*85];
        outputMap.put(this.outputOrder[2], this.rawres.out3); // float[1][inputSize/32][inputSize/32][3*85];
    }
    private int inference_elapsed;
    private int postprocess_elapsed;
    public String getLastElapsedTimeLog() {
        return String.format("inference: %dms postprocess: %dms", this.inference_elapsed, this.postprocess_elapsed);
    }
    public List<Recognition> runInference(){
        List<Recognition> bboxes = new ArrayList<>();
        long start = System.currentTimeMillis();
        this.tfliteInterpreter.runForMultipleInputsOutputs(inputArray, outputMap);
        long end = System.currentTimeMillis();
        this.inference_elapsed = (int)(end - start);

        //float[bbox_num][6]
        //                       (x1, y1, x2, y2, conf, class_idx)
        float[][] bbox_arrs = postprocess(
            this.rawres.out1,
            this.rawres.out2,
            this.rawres.out3,
            this.inputSize,
            this.conf_thresh,
            this.iou_thresh
        );
        long end2 = System.currentTimeMillis();
        this.postprocess_elapsed = (int)(end2 - end);
        for(float[] bbox_arr: bbox_arrs){
            bboxes.add(new Recognition(bbox_arr));
        }
        return bboxes;
    }
    static int[] coco80_to_91class_map = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
            64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90};
    static public int get_coco91_from_coco80(int idx){
        //assume idx < 80
        return coco80_to_91class_map[idx];
    }
    public void setConfThresh(float thresh){ this.conf_thresh = thresh;}
    public void setIoUThresh(float thresh) {this.iou_thresh = thresh;}

    //port from TfLite Object Detection example
    /** An immutable result returned by a Detector describing what was recognized. */
    public class Recognition {
        private final String[] coco_class_names = new String[]{"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
        private final Integer class_idx;
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        //private final String id;

        /** Display name for the recognition. */
        private final String title;

        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        private final Float confidence;

        /** Optional location within the source image for the location of the recognized object. */
        private RectF location;

        public Recognition(float[] bbox_array) {
            float x1 = bbox_array[0];
            float y1 = bbox_array[1];
            float x2 = bbox_array[2];
            float y2 = bbox_array[3];
            //this.id = (int)bbox_array[5];
            int class_id = (int)bbox_array[5];
            this.class_idx = class_id;
            this.title = coco_class_names[class_id];
            this.confidence = bbox_array[4];
            this.location = new RectF(x1, y1, x2, y2);
        }
        public Integer getClass_idx(){
            return class_idx;
        }
        /*public String getId() {
            return id;
        }*/

        public String getTitle() {
            return title;
        }

        public Float getConfidence() {
            return confidence;
        }

        public RectF getLocation() {
            return new RectF(location);
        }

        public void setLocation(RectF location) {
            this.location = location;
        }

        @Override
        public String toString() {
            String resultString = "";
            /*if (id != null) {
                resultString += "[" + id + "] ";
            }*/

            if (title != null) {
                resultString += title + " ";
            }

            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f);
            }

            if (location != null) {
                resultString += location + " ";
            }

            return resultString.trim();
        }
    }
}
