package org.hipi.tools.face;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.util.Tool;

public class FaceRecognitionHandler extends Configured implements Tool {

	@Override
	public int run(String[] args) throws Exception {

		Job job = Job.getInstance();
		Configuration conf = job.getConfiguration();

		if (FaceRecognition.run(args, job) == 1) {
			System.out.println("People List creation failed.");
			return 1;
		}

		String peopleListDir = conf.get("hipi.people.face.recognition.path");
		
		Path peopleListPath = new Path(peopleListDir);
	    FileSystem fileSystem = FileSystem.get(conf);
	    if (!fileSystem.exists(peopleListPath)) {
	      System.out.println("People List does not exist at location: " + peopleListPath);
	      System.exit(1);
	    }
	    
		// Run Training
		if (FaceRecognitionTraining.run(args, conf) == 1) {
			System.out.println("Neural Network training failed.");
			return 1;
		}

		// Indicate success
		return 0;
	}

}
