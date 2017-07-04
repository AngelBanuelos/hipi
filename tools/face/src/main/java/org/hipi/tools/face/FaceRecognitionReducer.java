package org.hipi.tools.face;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.hipi.opencv.OpenCVMatWritable;

public class FaceRecognitionReducer extends Reducer<Text, OpenCVMatWritable, Text, Text> {
	private Text result = new Text();
	private Text folder = new Text();
	private long id = 0L;

	@Override
	public void reduce(Text key, Iterable<OpenCVMatWritable> values, Context context)
			throws IOException, InterruptedException {
		
		int totalImagesPerFace = 0;
		//Grouping each key and counting all the occurrences.
		for (OpenCVMatWritable value : values) {
			totalImagesPerFace++;
		}
		result.set("Images found for " + key + "_" + id +" are :  " + totalImagesPerFace);
		//Saving the value in the given HDFS directory.
		context.write(key, result);
		
	}
}
