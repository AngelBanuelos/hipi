package org.hipi.tools.face;

import java.io.Serializable;

class AngelSerialized implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	String key;
	String value;
	
	protected AngelSerialized (String key, String value){
		this.key = new String(key);
		this.value = new String(value);
	}
	
	public String getKey() {
		return key;
	}
	
	public String getValue() {
		return value;
	}
	
}