package org.hipi.tools.face;

import static org.bytedeco.javacpp.opencv_imgproc.CV_RGB2GRAY;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.hipi.image.FloatImage;
import org.hipi.image.PixelArray;
import org.hipi.image.HipiImageHeader.HipiColorSpace;
import org.hipi.image.RasterImage;
import org.hipi.opencv.OpenCVUtils;

public class FaceUtils {

	public static boolean convertFloatImageToGrayscaleMat(RasterImage image, Mat cvImage) {

		// Convert FloatImage to Mat, and convert Mat to grayscale (if
		// necessary)
		HipiColorSpace colorSpace = image.getColorSpace();
		switch (colorSpace) {

		// if RGB, convert to grayscale
		case RGB:
			Mat cvImageRGB = OpenCVUtils.convertRasterImageToMat(image);
			opencv_imgproc.cvtColor(cvImageRGB, cvImage, CV_RGB2GRAY);
			return true;

		// if LUM, already grayscale
		case LUM:
			cvImage = OpenCVUtils.convertRasterImageToMat(image);
			return true;

		// otherwise, color space is not supported for this example. Skip input
		// image.
		default:
			System.out.println("HipiColorSpace [" + colorSpace + "] not supported in covar example. ");
			return false;
		}
	}

	
	public static void convertFloatImageToMat(RasterImage image, Mat cvImage) {
		cvImage = OpenCVUtils.convertRasterImageToMat(image);
	}

	/**
	 * Util method to convert RasterImage to Mat (JavaCPP or ByteDeco)
	 * @param image
	 * @return RGB Mat Image
	 */
	public static Mat convertFloatImageToMatManual(RasterImage image) {

		// Get pointer to image data
		FloatImage floatImage = ((FloatImage) image);
		float[] valData = floatImage.getData();

		int w = floatImage.getWidth();
		int h = floatImage.getHeight();

		Mat mat = new Mat(h, w, opencv_core.CV_8UC3);

		PixelArray pa = image.getPixelArray();
		double[] rgb = new double[w * h];
		for (int i = 0; i < w * h; i++) {
			int r = pa.getElemNonLinSRGB(i * 3 + 0);
			int g = pa.getElemNonLinSRGB(i * 3 + 1);
			int b = pa.getElemNonLinSRGB(i * 3 + 2);

			rgb[i] = (r << 16) | (g << 8) | b;
		}

		Mat tempMat = new Mat(rgb);
		tempMat.copyTo(mat);
		mat.convertTo(mat, opencv_core.CV_8UC3);

		return mat;

	}

	/**
	 * Util method to convert RasterImage to Mat (JavaCPP or ByteDeco)
	 * @param image
	 * @return RGB Mat image.
	 */
	public static Mat convertFloatImageToMatManualV2(RasterImage image) {
		
		int w = image.getWidth();
		int h = image.getHeight();

		BufferedImage bufferedImage = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);

		PixelArray pa = image.getPixelArray();
		int[] rgb = new int[w * h];
		for (int i = 0; i < w * h; i++) {

			int r = pa.getElemNonLinSRGB(i * 3 + 0);
			int g = pa.getElemNonLinSRGB(i * 3 + 1);
			int b = pa.getElemNonLinSRGB(i * 3 + 2);

			rgb[i] = (r << 16) | (g << 8) | b;
		}
		bufferedImage.setRGB(0, 0, w, h, rgb, 0, w);

		byte[] pixels = ((DataBufferByte) bufferedImage.getRaster().getDataBuffer()).getData();
		Mat mat = new Mat(pixels);
		return mat;
	}

	/**
	 * Util method to convert FloatImage to Mat (Native OpenCV)
	 * @param floatImage
	 * @return RGB Mat image.
	 */
	public static org.opencv.core.Mat convertFloatImageToOpenCVMat(FloatImage floatImage) {
		
		// Get dimensions of image
		int w = floatImage.getWidth();
		int h = floatImage.getHeight();
		// Get pointer to image data
		float[] valData = floatImage.getData();

		// Initialize 3 element array to hold RGB pixel average
		double[] rgb = { 0.0, 0.0, 0.0 };

		org.opencv.core.Mat mat = new org.opencv.core.Mat(h, w, opencv_core.CV_8UC3);

		// Traverse image pixel data in raster-scan order and update running
		// average
		for (int j = 0; j < h; j++) {
			for (int i = 0; i < w; i++) {
				rgb[0] = (double) valData[(j * w + i) * 3 + 0] * 255.0; // R
				rgb[1] = (double) valData[(j * w + i) * 3 + 1] * 255.0; // G
				rgb[2] = (double) valData[(j * w + i) * 3 + 2] * 255.0; // B
				mat.put(j, i, rgb);

			}
		}
		return mat;
	}

}
