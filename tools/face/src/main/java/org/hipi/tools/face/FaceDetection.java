package org.hipi.tools.face;

import org.hipi.imagebundle.mapreduce.HibInputFormat;

import java.net.URI;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.Job;

public class FaceDetection extends Configured implements Tool {

	private static final Options options = new Options();
	private static final Parser parser = (Parser) new BasicParser();

	static {
		options.addOption("f", "force", false, "force overwrite if output HIB already exists");
		options.addOption("h", "hdfs-input", false, "assume input directory is on HDFS");
		options.addOption("a", "action", true, "faceDetection FD, FaceRecognition FR, its a Must");
		options.addOption("m", "recognition-method", true,
				"LBPHFaceRecognizer = 1, FisherFaceRecognizer = 2," + " EigenFaceRecognizer = 3 ");
		options.addOption("mp", "image-limit-percentage", true,
				"Maximun number of images to be load per folder by percentage");
	}

	public int run(String[] args) throws Exception {

		// Check input arguments

		CommandLine line = null;
		String action = null;

		try {
			line = parser.parse(options, args);
		} catch (ParseException exp) {
			usage();
		}
		if (line == null || line.getArgs() == null || line.getArgs().length != 3) {
			usage();
		}
		boolean overwrite = false;
		if (line.hasOption("f")) {
			overwrite = true;
		}

		// Initialize and configure MapReduce job
		Job job = Job.getInstance();
		// Set input format class which parses the input HIB and spawns map
		// tasks
		job.setInputFormatClass(HibInputFormat.class);
		// Set the driver, mapper, and reducer classes which express the
		// computation
		job.setJarByClass(FaceDetection.class);
		job.setMapperClass(FaceDetectionMapperV2.class);
		job.setReducerClass(FaceDetectionReducer.class);
		// Set the types for the key/value pairs passed to/from map and reduce
		// layers
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(IntWritable.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);

		// Adding the HAAR-LIKE trained neural red.
		job.addCacheFile(new URI(args[0]));

		if (overwrite) {
			// configuration should contain reference to your namenode
			FileSystem fs = FileSystem.get(new Configuration());
			// true stands for recursively deleting the folder you gave
			fs.delete(new Path(args[2]), true);
		}
		
		// Set the input and output paths on the HDFS
		FileInputFormat.setInputPaths(job, new Path(args[1]));
		FileOutputFormat.setOutputPath(job, new Path(args[2]));

		// Create just one reduce task
		job.setNumReduceTasks(1);

		// Execute the MapReduce job and block until it completes
		boolean success = job.waitForCompletion(true);

		// Return success or failure
		return success ? 0 : 1;
	}

	private void usage() {
		System.out.println("Usage: FaceDetection <HAAR-LIKE cascade xml> <input HIB> <output directory>");
		System.exit(0);
	}

	public static void main(String[] args) throws Exception {
		ToolRunner.run(new FaceDetection(), args);
		System.exit(0);
	}

}
