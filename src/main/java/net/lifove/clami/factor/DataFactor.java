package net.lifove.clami.factor;

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
	
	public DataFactor computeValue() {
		return null;
	} 
	
	public double getValue() {
		return value;
	}
	
}
