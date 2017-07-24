package org.hipi.tools.face;

import java.io.IOException;
import java.util.ArrayList;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
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

		ArrayList<OpenCVMatWritable> images = new ArrayList<>();
		int totalImagesPerFace = 0;
		// Grouping each key and counting all the occurrences.
		for (OpenCVMatWritable value : values) {
			totalImagesPerFace++;
			images.add(value);
		}

		if (totalImagesPerFace == 0) {
			System.err.println("No images for people " + key.toString());
			return;
		}

		ArrayOpenCVMatWritable peopleImages = new ArrayOpenCVMatWritable();
		key = new Text(key + "_" + id++);

		peopleImages.setValues(images);

		peopleMap.put(key, peopleImages);

		Configuration conf = context.getConfiguration();
		String peopleListDir = conf.get("hipi.people.face.recognition.path");

		peopleMap.write(FileSystem.get(conf).create(new Path(peopleListDir)));

		context.write(NullWritable.get(), peopleMap);
	}

}
