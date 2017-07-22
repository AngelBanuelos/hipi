package org.hipi.tools.face;

import java.io.IOException;
import java.nio.IntBuffer;

import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.hipi.opencv.OpenCVMatWritable;

public class FaceRecognitionReducer extends Reducer<Text, OpenCVMatWritable, NullWritable, MapWritable> {

	private static volatile int id = 0;
	MapWritable peopleMap = new MapWritable();
	
	@Override
	public void reduce(Text key, Iterable<OpenCVMatWritable> values, Context context)
			throws IOException, InterruptedException {
		
		int totalImagesPerFace = 0;
		//Grouping each key and counting all the occurrences.
		for (OpenCVMatWritable value : values) {
			totalImagesPerFace++;
		}
		
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
		peopleMap.put(key, peopleImages);
		
		context.write(NullWritable.get(), peopleMap);
	}
}
