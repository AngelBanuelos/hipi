package org.hipi.tools.face;

import java.io.IOException;
import java.nio.IntBuffer;

import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.hipi.opencv.OpenCVMatWritable;

public class FaceRecognitionReducer extends Reducer<Text, OpenCVMatWritable, Text, MapWritable> {
//	private Text result = new Text();
//	private Text folder = new Text();
	private int id = 0;

	@Override
	public void reduce(Text key, Iterable<OpenCVMatWritable> values, Context context)
			throws IOException, InterruptedException {
		
		int totalImagesPerFace = 0;
		//Grouping each key and counting all the occurrences.
		for (OpenCVMatWritable value : values) {
			totalImagesPerFace++;
		}
		MapWritable peopleList = new MapWritable();
		ArrayWritable peopleImages = new ArrayWritable(OpenCVMatWritable.class);
		key = new Text(key + "_" + id++);
//		result.set("Images found for " + key +" are :  " + totalImagesPerFace);
		//Saving the value in the given HDFS directory.
//		MatVector images = new MatVector(totalImagesPerFace);
//		Mat labels = new Mat(totalImagesPerFace, 1, opencv_core.CV_32SC1);
//		IntBuffer labelsBuf = labels.createBuffer();
		
		int counter = 0;
		OpenCVMatWritable[] imageArray = new OpenCVMatWritable[totalImagesPerFace];
		for (OpenCVMatWritable value : values) {
			imageArray[counter] = value;
//			images.put(counter, value.getMat());
//			labelsBuf.put(counter, id);
			counter++;
		}
		peopleImages.set(imageArray);
		peopleList.put(key, peopleImages);
		
		context.write(new Text("AllPeople"), peopleList);
		
	}
}
