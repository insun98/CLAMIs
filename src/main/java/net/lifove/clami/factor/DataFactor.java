package net.lifove.clami.factor;

/**
 * This class is super class for data factors 
 */
public class DataFactor {

	String factorName ;
	double value;
	
	public DataFactor() {
		this.factorName = "";
		this.value = 0;
	}
	
	public DataFactor(String factorName, double value) {
		this.factorName = factorName;
		this.value = value;
	}
	
	/**
	 * Compute data factor value 
	 * @return
	 */
	public DataFactor computeValue() {
		return null;
	} 
	
	/**
	 * Return factor value 
	 * @return
	 */
	public double getValue() {
		return value;
	}
	
}
