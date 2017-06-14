package org.hipi.tools.face;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.RectVector;
import org.bytedeco.javacpp.opencv_objdetect;
import org.hipi.image.FloatImage;
import org.hipi.image.HipiImageHeader;

public class FaceDetectionMapper extends Mapper<HipiImageHeader, FloatImage, IntWritable, Text> {

	// Create a face detector from the cascade file in the resources
	// directory.
	private opencv_objdetect.CascadeClassifier faceDetector;

	public void map(HipiImageHeader key, FloatImage image, Context context) throws IOException, InterruptedException {

		// Verify that image was properly decoded, is of sufficient size, and
		// has three color channels (RGB)
		if (image != null && image.getWidth() > 1 && image.getHeight() > 1 && image.getNumBands() == 3) {
			
			int w = image.getWidth();
			int h = image.getHeight();

			Mat cvImage = new Mat(h, w, opencv_core.CV_8UC3);
			FaceUtils.convertFloatImageToGrayscaleMat(image, cvImage);
			cvImage.convertTo(cvImage, opencv_core.CV_8UC3);
			
			long faces = countFaces(cvImage);

			// String source = image.getMetaData("source");
			// String filename = image.getMetaData("filename");
			// long faces = detectFromHDFS(source.substring(source.length() -
			// 25) + "/" + filename);

			// Emit record to reducer
			context.write(new IntWritable(1), new Text("" + faces));

		}
	}

	public long detectFromHDFS(String hdfsPath) throws IOException {

		Configuration conf = new Configuration();

		FileSystem fs = FileSystem.get(conf);

		FileStatus[] files = fs.listStatus(new Path(hdfsPath));

		FSDataInputStream fdis = fs.open(files[0].getPath());

		BufferedInputStream bufferedInputStream = new BufferedInputStream(fdis);

		DataInputStream dis = new DataInputStream(new BufferedInputStream(bufferedInputStream));

		BufferedImage javaImage = ImageIO.read(dis);

		byte[] pixels = ((DataBufferByte) javaImage.getRaster().getDataBuffer()).getData();
		Mat mat = new Mat(pixels);

		return countFaces(mat);
	}

	public long countFaces(Mat image) {
		if (faceDetector == null || faceDetector.isNull()) {
			faceDetector = new opencv_objdetect.CascadeClassifier("./lbpcascade_frontalface.xml");
		}

		if (faceDetector == null || faceDetector.isNull()) {
			return 0;
		}
		RectVector faceDetections = new RectVector();
		if (faceDetector.address() != 0) {
			// Detect faces in the image.
			faceDetector.detectMultiScale(image, faceDetections);
		}
		return faceDetections.size();
	}

}