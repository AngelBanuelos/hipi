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
import org.hipi.image.FloatImage;
import org.hipi.image.HipiImageHeader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.objdetect.CascadeClassifier;

public class FaceDetectionMapperV2 extends Mapper<HipiImageHeader, FloatImage, Text, IntWritable> {

	static {
		//When using the native opencv lib is necessary to load the native lib first.
		
		System.load("/usr/local/share/OpenCV/java/libopencv_java300.so");
	}

	// Create a face detector from the cascade file in the resources
	// directory.
	private CascadeClassifier faceDetector;

	//Count faces.
	public int countFaces(Mat cvImage){
		if (faceDetector == null) {
			faceDetector = new CascadeClassifier("./lbpcascade_frontalface.xml");
		}
		if (faceDetector == null) {
			return 0;
		}
		
		MatOfRect faceDetections = new MatOfRect();
		if (faceDetector != null && !faceDetector.empty()) {
			// Detect faces in the image.
			faceDetector.detectMultiScale(cvImage, faceDetections);
		}
		return faceDetections.toArray().length;
	}

	public void map(HipiImageHeader key, FloatImage image, Context context) throws IOException, InterruptedException {

		// Verify that image was properly decoded, is of sufficient size, and
		// has three color channels (RGB)
		if (image != null && image.getWidth() > 1 && image.getHeight() > 1 && image.getNumBands() == 3) {
			
			Mat cvImage = FaceUtils.convertFloatImageToOpenCVMat(image);
			int faces = countFaces(cvImage);

			//To read images from hdfs using the signature in the hibimage
//			 String source = image.getMetaData("source");
//			String filename = image.getMetaData("filename");
//			 long faces = detectFromHDFS(source.substring(source.length() -
//			 25) + "/" + filename);
			
			//to count image per folder uncomment next line
//			Text folder = new Text(filename.substring(0, 7));
			Text folder = new Text("All");
			
			// Emit record to reducer
			context.write(folder, new IntWritable(faces));
		}
	}

	//Method to read images directly from HDFS converted 
	public long detectFromHDFS(String hdfsPath) throws IOException {
		
		Configuration conf = new Configuration();

		FileSystem fs = FileSystem.get(conf);
		
		FileStatus[] files = fs.listStatus(new Path(hdfsPath));

		FSDataInputStream fdis = fs.open(files[0].getPath());

		BufferedInputStream bufferedInputStream = new BufferedInputStream(fdis);

		DataInputStream dis = new DataInputStream(new BufferedInputStream(bufferedInputStream));

		BufferedImage javaImage = ImageIO.read(dis);

		byte[] pixels = ((DataBufferByte) javaImage.getRaster().getDataBuffer()).getData();
		Mat mat = new Mat();
		mat.put(0, 0, pixels);
		
		return countFaces(mat);

	}

}