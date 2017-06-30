package org.hipi.tools.face;

import java.net.URI;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.hipi.imagebundle.mapreduce.HibInputFormat;
import org.hipi.opencv.OpenCVMatWritable;

public class FaceRecognition extends Configured implements Tool {

	private static final Options options = new Options();
	private static final Parser parser = (Parser) new BasicParser();

	static {
		options.addOption("f", "force-method", false, " force training ");
		options.addOption("h", "hdfs-input", false, "assume input directory is on HDFS and is a Hib file");
		options.addOption("m", "recognition-method", true,
				"LBPHFaceRecognizer = 1, FisherFaceRecognizer = 2," + " EigenFaceRecognizer = 3 ");
	}

	private static void usage() {
		// usage
		HelpFormatter formatter = new HelpFormatter();
		formatter.printHelp("face.jar [options] <image directory HIB> <image(s) to predict>", options);

		System.exit(0);
	}

	@Override
	public int run(String[] args) throws Exception {

		CommandLine line = null;
		String action = null;

		try {
			line = parser.parse(options, args);
		} catch (ParseException exp) {
			usage();
		}
		if (line == null || line.getArgs() == null || line.getArgs().length != 2) {
			usage();
		}
		boolean overwrite = false;
		if (line.hasOption("f")) {
			overwrite = true;
		}
		args = line.getArgs();

		// Initialize and configure MapReduce job
		Job job = Job.getInstance();
		// Set input format class which parses the input HIB and spawns map
		// tasks
		job.setInputFormatClass(HibInputFormat.class);
		// Set the driver, mapper, and reducer classes which express the
		// computation
		job.setJarByClass(FaceRecognition.class);
		job.setMapperClass(FaceRecognitionMapper.class);
		job.setReducerClass(FaceRecognitionReducer.class);
		// Set the types for the key/value pairs passed to/from map and reduce
		// layers
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(OpenCVMatWritable.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);

		if (overwrite) {
			// configuration should contain reference to your namenode
			FileSystem fs = FileSystem.get(new Configuration());
			// true stands for recursively deleting the folder you gave
			fs.delete(new Path(args[1]), true);
		}
		
		// Set the input and output paths on the HDFS
		FileInputFormat.setInputPaths(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));

		// Create just one reduce task
		job.setNumReduceTasks(1);

		// Execute the MapReduce job and block until it completes
		boolean success = job.waitForCompletion(true);

		// Return success or failure
		return success ? 0 : 1;
	}

}
