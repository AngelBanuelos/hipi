package org.hipi.tools.face;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.SerializationUtils;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.IntWritable;
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
		peopleMap.put(key, peopleImages);
		FileUtils.writeByteArrayToFile(new File("/tmp/test8/people-output/AngelSerialized"), SerializationUtils.serialize(new AngelSerialized("Angel_Key","Angel_Value")));
		context.write(NullWritable.get(), peopleMap);
	}
	
	class AngelSerialized implements Serializable {
		/**
		 * 
		 */
		private static final long serialVersionUID = 1L;
		
		Text key;
		Text value;
		
		protected AngelSerialized (String key, String value){
			this.key = new Text(key);
			this.value = new Text(value);
		}
		
		public Text getKey() {
			return key;
		}
		
		public Text getValue() {
			return value;
		}
		
	}
}
