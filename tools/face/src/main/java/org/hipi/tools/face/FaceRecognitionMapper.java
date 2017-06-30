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

public class FaceRecognitionMapper extends Mapper<HipiImageHeader, FloatImage, Text, IntWritable> {

	public void map(HipiImageHeader key, FloatImage image, Context context) throws IOException, InterruptedException {

		
		Mat faceRecognition = new Mat(image.getHeight(), image.getWidth(), opencv_core.CV_32FC1);
		FaceUtils.convertFloatImageToGrayscaleMat(image, faceRecognition);
		Text fileName = new Text(key.getMetaData("filename").split("\\-")[0]);
//		ImageContainer img = new ImageContainer(key, image);
		
//		context.write(fileName, new OpenCVMatWritable(faceRecognition));
		context.write(fileName, new IntWritable(1));
	}
}
