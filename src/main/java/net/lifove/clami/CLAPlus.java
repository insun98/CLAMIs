package net.lifove.clami;

import net.lifove.clami.util.Utils;
import weka.core.Instances;

/**
 * This class run for CLA+. 
 */
public class CLAPlus extends CLA implements ICLA {

	/**
	 * Get CLA result
	 * @param instances
	 * @param percentileCutoff; cutoff percentile for cluster
	 * @param positiveLabel; string value of positive label 
	 * @param supress detailed prediction results
	 * @param filePath; string value of file name  
	 */
	public void getResult(Instances instances, double percentileCutoff, String positiveLabel, boolean suppress, String filePath) {
		getResult(instances, percentileCutoff, positiveLabel, suppress, false, filePath);
	}

	/**
	 * Get CLA result
	 * @param instances
	 * @param percentileCutoff; cutoff percentile for cluster
	 * @param positiveLabel; string value of positive label 
	 * @param supress detailed prediction results
	 * @param experimental; boolean value whether experimental or not 
	 * @param filePath; string value of file name  
	 */
	public void getResult(Instances instances, double percentileCutoff, String positiveLabel, boolean suppress,
			boolean experimental, String filePath) {

			instancesByCLA = clustering(instances, percentileCutoff, positiveLabel);
		printResult(instances, experimental, filePath, suppress, positiveLabel);
	}

	/**
	 * Cluster with percentileCutoff. Set class value to positive if K is higher than cutoff of cluster.
	 * @param instances
	 * @param percentileCutoff; cutoff percentile for cluster 
	 * @param positiveLabel; string value of positive label 
	 */
	@Override
	public Instances clustering(Instances instances, double percentileCutoff, String positiveLabel) {

		Instances instancesByCLA = new Instances(instances); 
		double[] cutoffsForHigherValuesOfAttribute = Utils.getHigherValueCutoffs(instances, percentileCutoff); 
		Double[] K = new Double[instances.numInstances()]; 

		for (int instIdx = 0; instIdx < instances.numInstances(); instIdx++) {
			K[instIdx] = 0.0;
			Double sum = 0.0;
			for (int attrIdx = 0; attrIdx < instances.numAttributes(); attrIdx++) {
				if (attrIdx == instances.classIndex())
					continue;
				Double violationDegree = 1 / (1 + Math.pow(Math.E,
						-(instances.get(instIdx).value(attrIdx) - cutoffsForHigherValuesOfAttribute[attrIdx])));
				sum = sum + violationDegree;
				
			}
			K[instIdx] = sum / instances.numAttributes();
		}

		double cutoffOfKForTopClusters = 0.5;

		for (int instIdx = 0; instIdx < instances.numInstances(); instIdx++) {
			if (K[instIdx] >= cutoffOfKForTopClusters)
				instancesByCLA.instance(instIdx).setClassValue(positiveLabel);
			else
				instancesByCLA.instance(instIdx).setClassValue(Utils.getNegLabel(instancesByCLA, positiveLabel));
		}
		return instancesByCLA;

	}

}