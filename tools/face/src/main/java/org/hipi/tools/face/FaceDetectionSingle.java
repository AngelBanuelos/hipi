package org.hipi.tools.face;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacpp.opencv_objdetect;

/**
 *
 * @author angel_banuelos
 */
public class FaceDetectionSingle {

    private opencv_objdetect.CascadeClassifier faceDetector;
    
    public long countFaces(Mat image, String [] args) {

    	if (faceDetector == null) {
            faceDetector = new opencv_objdetect.CascadeClassifier(args[1]);
        }
        if (faceDetector == null) {
            return -1;
        }
        // Detect faces in the image.
        opencv_core.RectVector faceDetections = new opencv_core.RectVector();
        faceDetector.detectMultiScale(image, faceDetections);

        return faceDetections.size();
    }
    
    public void setup(String args []){
         Mat testImage = opencv_imgcodecs.imread(args[0]);
         long num  = this.countFaces(testImage, args);
         System.out.println("Num " + num);
    
    }
    
    public static void main(String[] args) {
    	
    	FaceDetectionSingle face = new FaceDetectionSingle();
        face.setup(args);
        
    }

}