package org.hipi.tools.face;

import java.util.List;

import org.apache.hadoop.io.ArrayWritable;
import org.hipi.opencv.OpenCVMatWritable;

public class ArrayOpenCVMatWritable extends ArrayWritable {

	private OpenCVMatWritable[] values;

	public ArrayOpenCVMatWritable() {
		super(OpenCVMatWritable.class);
	}

	public void setValues(List<OpenCVMatWritable> values) {
		if (values == null || values.isEmpty()) {
			return;
		}

		int counter = 0;
		OpenCVMatWritable[] imageArray = new OpenCVMatWritable[values.size()];

		for (OpenCVMatWritable value : values) {
			imageArray[counter] = new OpenCVMatWritable(value.getMat());
			counter++;
		}

		this.values = imageArray;
		set(imageArray);
	}

}
