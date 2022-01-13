package net.lifove.clami.util;

public class DataFeatures {

	private double percentileCutoff;
	private String positiveLabel;
	private int numOfInstances;
	private int numOfMetrics;
	private double originalEPV;
	private double reverseEPV;
	private int numOfGroups;
	private double IGR;
	private double GIR;
	private int valueOfMaxVote;
	private int numOfMaxVote;

	public DataFeatures(int numOfInstances, int numOfMetrics, double percentileCutoff, String positiveLabel) {
		this.numOfInstances = numOfInstances;
		this.numOfMetrics = numOfMetrics;
		this.originalEPV = numOfInstances/numOfMetrics;
		this.reverseEPV = numOfMetrics/numOfInstances;
		this.percentileCutoff = percentileCutoff;
		this.positiveLabel = positiveLabel;
	}

	// get from Tools.ksTest() 
	public void setNumOfGroups(int numOfGroups) {
		this.numOfGroups = numOfGroups;
		this.IGR = numOfInstances/numOfGroups;
		this.GIR = numOfGroups/numOfInstances;
	}
	public void setValueOfMaxVote(int valueOfMaxVote) {
		this.valueOfMaxVote = valueOfMaxVote;
	}
	public void setNumOfMaxVote(int numOfMaxVote) {
		this.numOfMaxVote = numOfMaxVote;
	}
	
	public void printAllFeatures() {
		
		System.out.println("number of instances: " + numOfInstances);
		System.out.println("number of metrics: " + numOfMetrics);
		System.out.println("original EPV: " + originalEPV);
		System.out.println("reverse EPV: " + reverseEPV);
		System.out.println("number of groups: " + numOfGroups);
		
		System.out.println("IGR (#instances/#groups): " + IGR);
		System.out.println("GIR (#groups/#instances): " + GIR);
		
		System.out.println("value of max vote" + valueOfMaxVote);
		System.out.println("number of max vote: " + numOfMaxVote);
		
	}
	
}