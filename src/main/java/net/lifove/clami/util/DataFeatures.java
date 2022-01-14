package net.lifove.clami.util;

public class DataFeatures {

	private int numOfInstances;
	private int numOfMetrics;
	private double originalEPV;
	private double reverseEPV;
	private int numOfGroups;
	private double IGR;
	private double GIR;
	private int valueOfMaxVote;
	private int numOfMaxVote;

	/**
	 * Initialize all data evaluation metric 
	 * @param numOfInstances
	 * @param numOfMetrics
	 * @param numOfGroups
	 * @param valueOfMaxVote
	 * @param numOfMaxVote
	 */
	public DataFeatures(int numOfInstances, int numOfMetrics, int numOfGroups, int valueOfMaxVote, int numOfMaxVote) {
		this.numOfInstances = numOfInstances;
		this.numOfMetrics = numOfMetrics;
		this.originalEPV = (double) numOfInstances/numOfMetrics;
		this.reverseEPV = (double) numOfMetrics/numOfInstances;
		
		this.numOfGroups = numOfGroups;
		this.IGR = (double) numOfInstances/numOfGroups;
		this.GIR = (double) numOfGroups/numOfInstances;
		
		this.valueOfMaxVote = valueOfMaxVote;
		this.numOfMaxVote = numOfMaxVote;
	}

	/**
	 * Print all features 
	 */
	public void printAllFeatures() {
		
		System.out.println("===== Features =====");
		System.out.println("number of instances: " + numOfInstances);
		System.out.println("number of metrics: " + numOfMetrics);
		System.out.println("original EPV: " + originalEPV);
		System.out.println("reverse EPV: " + reverseEPV);
		System.out.println("number of groups: " + numOfGroups);
		
		System.out.println("IGR (#instances/#groups): " + IGR);
		System.out.println("GIR (#groups/#instances): " + GIR);
		
		System.out.println("value of max vote: " + valueOfMaxVote);
		System.out.println("number of max vote: " + numOfMaxVote);
		System.out.println("====================");
		
	}
	
}