package org.hipi.tools.face;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;

public class FaceRecognitionReducer extends Reducer<Text, IntWritable, Text, Text> {
	private Text result = new Text();
	private Text folder = new Text();

	@Override
	public void reduce(Text key, Iterable<IntWritable> values, Context context)
			throws IOException, InterruptedException {
		
		int totalImagesPerFace = 0;
		int size = 0;
		//Grouping each key and counting all the occurrences.
		for (IntWritable value : values) {
			totalImagesPerFace += value.get();
			size++;
		}
		result.set("Images found for " + key + " are :  " + totalImagesPerFace);
		//Saving the value in the given HDFS directory.
		context.write(key, result);
		
	}
}
