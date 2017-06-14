package org.hipi.tools.face;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.bytedeco.javacpp.opencv_core.Size;
import org.hipi.image.FloatImage;

public class FaceDetectionReducer extends Reducer<Text, IntWritable, Text, Text> {
	private Text result = new Text();
	private Text folder = new Text();

	@Override
	public void reduce(Text key, Iterable<IntWritable> values, Context context)
			throws IOException, InterruptedException {
		
		int totalFoundFaces = 0;
		int size = 0;
		//Grouping each key and counting all the occurrences.
		for (IntWritable value : values) {
			totalFoundFaces += value.get();
			size++;
		}
		result.set("Analyzed Images " + size + " Found faces " + totalFoundFaces);
		//Saving the value in the given HDFS directory.
		context.write(key, result);
		
	}

}
