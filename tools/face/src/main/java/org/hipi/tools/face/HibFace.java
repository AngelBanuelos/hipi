package org.hipi.tools.face;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.Parser;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.util.ToolRunner;

public class HibFace extends Configured {

	private static final Options options = new Options();
	private static final Parser parser = (Parser) new BasicParser();

	static {
		options.addOption("f", "force", false, "force overwrite if output HIB already exists");
		options.addOption("h", "hdfs-input", false, "assume input directory is on HDFS");
		options.addOption("a", "action", true, "faceDetection FD, FaceRecognition FR, FaceRecognitionSingle thread NonFR,  its a Must");
		options.addOption("m", "recognition-method", true,
				"LBPHFaceRecognizer = 1, FisherFaceRecognizer = 2," + " EigenFaceRecognizer = 3 ");
		options.addOption("mp", "image-limit-percentage", true,
				"Maximun number of images to be load per folder by percentage");
	}

	private static void usage() {
		// usage
		HelpFormatter formatter = new HelpFormatter();
		formatter.printHelp("face.jar [options] <args>", options);
		System.exit(0);
	}

	public static void main(String[] args) throws Exception {

		CommandLine line = null;
		String action = null;

		try {
			line = parser.parse(options, args);
		} catch (ParseException exp) {
			usage();
		}
		if (line == null) {
			usage();
		}
		if (!line.hasOption("a")) {
			usage();
		} else {
			action = line.getOptionValue("a");
		}
		if (action != null && action.equalsIgnoreCase("FD")) {
			ToolRunner.run(new FaceDetection(), args);
		} else if (action != null && action.equalsIgnoreCase("FR")) {
			ToolRunner.run(new FaceRecognition(), args);
		} else if (action != null && action.equalsIgnoreCase("NonFR")) {
			FaceRecognitionSingle.main(args);
		} else {
			usage();
		}
		
		System.exit(0);
	}

}
