package org.hipi.tools.face;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.hipi.image.FloatImage;
import org.hipi.image.HipiImageHeader;
import org.hipi.opencv.OpenCVMatWritable;

public class FaceRecognitionMapper extends Mapper<HipiImageHeader, FloatImage, IntWritable, OpenCVMatWritable> {

	public void map(HipiImageHeader key, FloatImage image, Context context) throws IOException, InterruptedException {

		
		Mat faceRecognition = new Mat(image.getHeight(), image.getWidth(), opencv_core.CV_32FC1);
		FaceUtils.convertFloatImageToGrayscaleMat(image, faceRecognition);
		
		context.write(new IntWritable(0), new OpenCVMatWritable(faceRecognition));

	}

}
