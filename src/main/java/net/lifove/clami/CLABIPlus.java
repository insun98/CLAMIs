package net.lifove.clami;

import weka.core.Instances;

public class CLABIPlus extends CLABI implements ICLAMI {

	/**
	 * Constructor
	 * @param mlAlg: machine learning algorithm
	 * @param isExperimental: to check if experiment option is on
	 */
	CLABIPlus(String mlAlg, boolean isExperimental) {
		super(mlAlg, isExperimental);
		// TODO Auto-generated constructor stub
	}
	
	/**
	 * Get CLABI result
	 * @param instances
	 * @param percentileCutoff cutoff percentile for top and bottom clusters
	 * @param positiveLabel positive label string value
	 * @param suppress detailed prediction results
	 * @param filePath: name of the running file
	 * @return instances labeled by CLAMI
	 */
	public void getResult(Instances instances, double percentileCutoff, String positiveLabel, boolean suppress, String filePath) {
		getResult(instances, percentileCutoff, positiveLabel, suppress, false, filePath);
	}
	
	/**
	 * To do clustering
	 * @param instances
	 * @param percentileCutoff cutoff percentile for top and bottom clusters
	 * @param positiveLabel positive label string value
	 */
	@Override
	public Instances clustering(Instances instances, double percentileCutoff, String positiveLabel) {
		CLAPlus claPlus = new CLAPlus();
		instancesByCLA = claPlus.clustering(instances, percentileCutoff, positiveLabel);
		return null;
	}
	
}
