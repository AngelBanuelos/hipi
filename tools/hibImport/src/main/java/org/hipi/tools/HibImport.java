package org.hipi.tools;

import org.hipi.imagebundle.HipiImageBundle;
import org.hipi.image.HipiImageHeader.HipiImageFormat;

import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.Parser;
import org.apache.commons.cli.ParseException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FSDataInputStream;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Arrays;

public class HibImport {

	private static final Options options = new Options();
	private static final Parser parser = (Parser) new BasicParser();
	// This variables are set to -1 to indicated that will upload all images and
	// all folders
	private static int folderLimit = -1;
	private static int imageQtyLimit = -1;
	private static int imageQtyLimitByPercentage = 100;

	static {
		options.addOption("f", "force", false, "force overwrite if output HIB already exists");
		options.addOption("h", "hdfs-input", false, "assume input directory is on HDFS");
		options.addOption("l", "folder-limit", true, "Maximum number of folders to be load per HDFS file");
		options.addOption("m", "image-limit", true, "Maximum number of images to be load per folder");
		options.addOption("mp", "image-limit-percentage", true,
				"Maximun number of images to be load per folder by percentage");
	}

	private static void usage() {
		// usage
		HelpFormatter formatter = new HelpFormatter();
		formatter.printHelp("hibImport.jar [options] <image directory> <output HIB>", options);

		System.exit(0);
	}

	public static void main(String[] args) throws IOException {

		// Attempt to parse the command line arguments
		CommandLine line = null;
		try {
			line = parser.parse(options, args);
		} catch (ParseException exp) {
			usage();
		}
		if (line == null) {
			usage();
		}

		String[] leftArgs = line.getArgs();
		if (leftArgs.length != 2) {
			usage();
		}

		boolean hdfsInput = false;
		if (line.hasOption("h")) {
			hdfsInput = true;
		}

		String imageDir = leftArgs[0];
		String outputHib = leftArgs[1];

		boolean overwrite = false;
		if (line.hasOption("f")) {
			overwrite = true;
		}

		if (line.hasOption("l")) {
			String fldLmt = line.getOptionValue("l");
			folderLimit = Integer.parseInt(fldLmt);
		}

		if (line.hasOption("m")) {
			String imgLmt = line.getOptionValue("m");
			imageQtyLimit = Integer.parseInt(imgLmt);
		}

		if (line.hasOption("mp")) {
			String imgLmtPrctg = line.getOptionValue("mp");
			imageQtyLimitByPercentage = Integer.parseInt(imgLmtPrctg);
		}

		System.out.println("Input image directory: " + imageDir);
		System.out.println("Input FS: " + (hdfsInput ? "HDFS" : "local FS"));
		System.out.println("Output HIB: " + outputHib);
		System.out.println("Overwrite HIB if it exists: " + (overwrite ? "true" : "false"));

		Configuration conf = new Configuration();
		System.out.println("Conf created");
		FileSystem fs = FileSystem.get(conf);
		System.out.println("fs created");
		int imageAdded = 0;
		if (hdfsInput) {

			System.out.println("hdfsInput is true");

			FileStatus[] files = fs.listStatus(new Path(imageDir));
			if (files == null) {
				System.err.println(String.format("Did not find any files in the HDFS directory [%s]", imageDir));
				System.exit(0);
			}
			System.out.println("Files/Directorosis" + files.length);
			Arrays.sort(files);
			System.out.println("Sorted" + files.length);
			HipiImageBundle hib = new HipiImageBundle(new Path(outputHib), conf);
			hib.openForWrite(overwrite);
			System.out.println(outputHib + " HIB CREATED");

			int folderQuantity = folderLimit;
			for (FileStatus file : files) {

				if (folderQuantity == 0) {
					break;
				}

				String source = file.getPath().toString();

				HashMap<String, String> metaData = new HashMap<String, String>();
				metaData.put("source", source);
				String fileName = file.getPath().getName();

				if (file.isDirectory()) {
					FileStatus[] files2 = fs.listStatus(new Path(imageDir + "/" + fileName + "/"));
					if (files2 == null) {
						System.err.println(String.format("Did not find any files in the HDFS directory [%s]",
								imageDir + "/" + fileName));
						continue;
					}
					Arrays.sort(files2);
					int imageNumPerFolder = imageQtyLimit;
					if (imageQtyLimitByPercentage < 100 && imageQtyLimitByPercentage > 0) {
						int total = files2.length;
						imageNumPerFolder = (total * imageQtyLimitByPercentage / 100);
						System.out.println(fileName + " images to load: " + imageNumPerFolder + " of " + total);
					}
					for (FileStatus file2 : files2) {
						if (imageNumPerFolder == 0) {
							break;
						}
						HashMap<String, String> metaData2 = new HashMap<String, String>();
						metaData2.put("source", source);

						FSDataInputStream fdis2 = fs.open(file2.getPath());
						String fileName2 = file2.getPath().getName();

						String suffix = fileName2.substring(fileName2.lastIndexOf('.'));
						metaData2.put("filename", fileName + "-" + "RCG_" + fileName2);

						if (suffix.compareTo(".jpg") == 0 || suffix.compareTo(".jpeg") == 0) {
							hib.addImage(fdis2, HipiImageFormat.JPEG, metaData2);
							imageAdded++;
						} else if (suffix.compareTo(".png") == 0) {
							hib.addImage(fdis2, HipiImageFormat.PNG, metaData2);
							imageAdded++;
						}
						imageNumPerFolder--;
					}

				} else {
					FSDataInputStream fdis = fs.open(file.getPath());
					String suffix = fileName.substring(fileName.lastIndexOf('.'));
					metaData.put("filename", fileName);
					if (suffix.compareTo(".jpg") == 0 || suffix.compareTo(".jpeg") == 0) {
						hib.addImage(fdis, HipiImageFormat.JPEG, metaData);
						imageAdded++;
						System.out.println(" ** added: " + fileName);
					} else if (suffix.compareTo(".png") == 0) {
						hib.addImage(fdis, HipiImageFormat.PNG, metaData);
						imageAdded++;
						System.out.println(" ** added: " + fileName);
					}
				}
				folderQuantity--;
			}

			hib.close();

		} else {

			File folder = new File(imageDir);
			File[] files = folder.listFiles();
			Arrays.sort(files);

			if (files == null) {
				System.err.println(String.format("Did not find any files in the local FS directory [%s]", imageDir));
				System.exit(0);
			}

			HipiImageBundle hib = new HipiImageBundle(new Path(outputHib), conf);
			hib.openForWrite(overwrite);

			for (File file : files) {
				FileInputStream fis = new FileInputStream(file);
				String localPath = file.getPath();
				HashMap<String, String> metaData = new HashMap<String, String>();
				metaData.put("source", localPath);
				String fileName = file.getName();
				metaData.put("filename", fileName);
				String suffix = fileName.substring(fileName.lastIndexOf('.'));
				if (suffix.compareTo(".jpg") == 0 || suffix.compareTo(".jpeg") == 0) {
					hib.addImage(fis, HipiImageFormat.JPEG, metaData);
					System.out.println(" ** added: " + fileName);
				} else if (suffix.compareTo(".png") == 0) {
					hib.addImage(fis, HipiImageFormat.PNG, metaData);
					System.out.println(" ** added: " + fileName);
				}
			}

			hib.close();

		}

		System.out.println("Created: " + outputHib + " and " + outputHib + ".dat" + 
									" images " + imageAdded + " added");
	}

}
