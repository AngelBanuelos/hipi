package org.hipi.tools.face;

import java.io.File;
import java.io.IOException;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.List;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.apache.commons.cli.BasicParser;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_face;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.hipi.image.FloatImage;
import org.hipi.image.HipiImageFactory;
import org.hipi.image.HipiImageHeader;
import org.hipi.image.RasterImage;
import org.hipi.imagebundle.HipiImageBundle;

public class FaceRecognitionSingle {

	private static final Options options = new Options();
	private static final Parser parser = (Parser) new BasicParser();

	private static int recognizerMethod = 1;
	private static String saveLocation = "";
	private static boolean faceRecognizerLoaded = false;
	private static boolean forceTrainig = false;
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

	private static FaceRecognizer faceRecognizer = null;

	public HipiImageBundle openHib(String trainingDir) {
		HipiImageBundle hib = null;
		try {
			hib = new HipiImageBundle(new Path(trainingDir), new Configuration(),
					HipiImageFactory.getFloatImageFactory());
			System.out.println("DIR >>>>" + hib.getPath().getName());
			configFRG(hib.getPath().getName());
			hib.openForRead(0);
			return hib;
		} catch (Exception ex) {
			System.err.println(ex.getMessage());
			ex.printStackTrace();
			System.exit(0);
			return null;
		}
	}

	private void configFRG(String name) {
		// Test
		switch (recognizerMethod) {
		case 1:
			System.out.println("opencv_face.createLBPHFaceRecognizer()");
			faceRecognizer = opencv_face.createLBPHFaceRecognizer();
			saveLocation = "/root/hipi/" + name + ".lbph.predict.opencv";
			break;
		case 2:
			System.out.println("opencv_face.createFisherFaceRecognizer()");
			faceRecognizer = opencv_face.createFisherFaceRecognizer();
			saveLocation = "/root/hipi/" + name + ".fisher.predict.opencv";
			break;
		case 3:
			System.out.println("opencv_face.createEigenFaceRecognizer()");
			faceRecognizer = opencv_face.createEigenFaceRecognizer();
			saveLocation = "/root/hipi/" + name + ".eigen.predict.opencv";
			break;
		default:
			System.err.println("Method do not exists");
			System.exit(0);
		}

		if (!forceTrainig && new File(saveLocation).exists()) {
			faceRecognizer.load(saveLocation);
			faceRecognizerLoaded = true;
			return;
		}
	}

	public void train(HipiImageBundle hib) throws IOException {
		if (faceRecognizerLoaded) {
			System.out.println("Neural Net Already Trained");
			return;
		}

		int count = 0;
		List<ImageContainer> imageContainerList = new ArrayList<>();

		while (hib.next()) {
			try {
				if (hib != null) {
					ImageContainer img = new ImageContainer(hib.currentHeader(), (FloatImage) hib.currentImage());
					imageContainerList.add(img);
					count++;
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

		MatVector images = new MatVector(count);

		Mat labels = new Mat(count, 1, opencv_core.CV_32SC1);
		IntBuffer labelsBuf = labels.createBuffer();

		int counter = 0;
		System.out.println("Loading images and labels");
		for (ImageContainer imageContainer : imageContainerList) {

			Mat img = new Mat(imageContainer.image.getHeight(), imageContainer.image.getWidth(), opencv_core.CV_32FC1);

			FaceUtils.convertFloatImageToGrayscaleMat(imageContainer.image, img);
			int label = Integer.parseInt(imageContainer.header.getMetaData("filename").split("\\-")[0]);

			images.put(counter, img);
			labelsBuf.put(counter, label);

			counter++;
			System.out.print("Image num: " + counter + "\r");
		}
		System.out.println("\nload finish");
		try {

			System.out.println("training");
			faceRecognizer.train(images, labels);
			System.out.println("trained");

			System.out.println("saving training");
			faceRecognizer.save(saveLocation);
			System.out.println(saveLocation);

		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			hib.close();
		}
	}

	// public int predict(RasterImage floatImage) {
	// Mat testImage = null;
	// FaceUtils.convertFloatImageToGrayscaleMat(floatImage, testImage);
	// int predictedLabel = faceRecognizer.predict(testImage);
	// return predictedLabel;
	// }
	//
	// public int predict(Mat image) {
	// // Mat imageRGB = image;
	// // opencv_imgproc.cvtColor(imageRGB, image, CV_RGB2GRAY);
	// int predictedLabel = faceRecognizer.predict(image);
	// return predictedLabel;
	// }

	public int predict(RasterImage floatImage) {
		Mat testImage = null;
		FaceUtils.convertFloatImageToGrayscaleMat(floatImage, testImage);
		int predictedLabel = faceRecognizer.predict_label(testImage);
		return predictedLabel;
	}

	public int predict(Mat image) {
		// Mat imageRGB = image;
		// opencv_imgproc.cvtColor(imageRGB, image, CV_RGB2GRAY);
		int predictedLabel = faceRecognizer.predict_label(image);
		return predictedLabel;
	}

	class ImageContainer {
		private HipiImageHeader header;
		private RasterImage image;

		ImageContainer(HipiImageHeader header, RasterImage image) {
			this.header = header;
			this.image = image;
		}

		public RasterImage getImage() {
			return image;
		}

		public void setImage(RasterImage image) {
			this.image = image;
		}

		public HipiImageHeader getHeader() {
			return header;
		}

		public void setHeader(HipiImageHeader header) {
			this.header = header;
		}

	}

	public static void main(String[] args) throws Exception {
		FaceRecognitionSingle faceRecognitionMaper = new FaceRecognitionSingle();
		// Attempt to parse the command line arguments
		CommandLine line = null;
		try {
			line = parser.parse(options, args);
		} catch (ParseException exp) {
			exp.printStackTrace();
			usage();
		}
		System.out.println("line  " + line);
		if (line == null) {
			usage();
		}

		System.out.println("Args " + line.getArgs());
		
		String[] leftArgs = line.getArgs();
		if (leftArgs.length != 2) {
			usage();
		}

		boolean hdfsInput = false;
		if (line.hasOption("h")) {
			hdfsInput = true;
		}

		if (line.hasOption("f")) {
			forceTrainig = true;
		}

		if (line.hasOption("m")) {
			String method = line.getOptionValue("m");
			recognizerMethod = Integer.parseInt(method);
			if (recognizerMethod < 0 || recognizerMethod > 3) {
				throw new Exception("Method has not been programmed");
			}
		}

		String imageDir = leftArgs[0];
		String testingDirOrImage = leftArgs[1];

		try {

			if (hdfsInput) {
				HipiImageBundle hib = faceRecognitionMaper.openHib(imageDir);
				// Training
				faceRecognitionMaper.train(hib);
			} else {
				System.err.println("System location not yet implemented");
				System.exit(1);
			}

			// Prediction
			System.out.println("Predicting ");
			Mat testImage = opencv_imgcodecs.imread(testingDirOrImage, opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
			int predict = faceRecognitionMaper.predict(testImage);
			System.out.println("Prediected " + predict);

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
