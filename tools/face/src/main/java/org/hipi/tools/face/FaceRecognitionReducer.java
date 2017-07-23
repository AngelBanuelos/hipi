package org.hipi.tools.face;

import java.io.IOException;

import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
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
		
		int counter = 0;
		OpenCVMatWritable[] imageArray = new OpenCVMatWritable[totalImagesPerFace];
		for (OpenCVMatWritable value : values) {
			imageArray[counter] = value;
			counter++;
		}
		peopleImages.set(imageArray);
		peopleMap.put(key, new Text("Angel Test"));
		
		context.write(NullWritable.get(), peopleMap);
	}
}
