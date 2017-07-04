package org.hipi.tools.face;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.hipi.image.FloatImage;
import org.hipi.image.HipiImageHeader;
import org.hipi.opencv.OpenCVMatWritable;
import org.hipi.tools.face.FaceRecognitionSingle.ImageContainer;

public class FaceRecognitionMapper extends Mapper<HipiImageHeader, FloatImage, Text, OpenCVMatWritable> {

	public void map(HipiImageHeader key, FloatImage image, Context context) throws IOException, InterruptedException {
		Mat faceRecognition = null;
		if (image != null) {
			faceRecognition = new Mat(image.getHeight(), image.getWidth(), opencv_core.CV_32FC1);
			FaceUtils.convertFloatImageToGrayscaleMat(image, faceRecognition);
		}
		Text fileName = new Text(key.getMetaData("filename").split("\\-")[0]);
		// ImageContainer img = new ImageContainer(key, image);
		if (faceRecognition == null) {
			System.err.println(" fileName error: " + fileName);
			faceRecognition = new Mat();
		}
		context.write(fileName, new OpenCVMatWritable(faceRecognition));
	}
}
