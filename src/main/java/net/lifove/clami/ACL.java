package net.lifove.clami;

import net.lifove.clami.util.Utils;
import weka.core.Instances;

/**
 * This class run for ACL. 
 */
public class ACL implements ICLA {
	protected Instances instancesByCLA = null;

	/**
	 * Get ACL result
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
	 * Get ACL result
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
		//instances = removeNoiseMetrics(instances);
		
	}
	

	/**
	 * Cluster with percentileCutoff. Set class value to positive if K is higher than cutoff of cluster.
	 * @param instances
	 * @param percentileCutoff; cutoff percentile for cluster 
	 * @param positiveLabel; string value of positive label 
	 */
	public Instances clustering(Instances instances, double percentileCutoff, String positiveLabel) {

		instancesByCLA = new Instances(instances);
		double[] cutoffsForHigherValuesOfAttribute = Utils.getHigherValueCutoffsForACL(instances);
		
		double[] K = new double[instances.numInstances()];

		for (int instIdx = 0; instIdx < instances.numInstances(); instIdx++) {
			K[instIdx] = 0.0;

			for (int attrIdx = 0; attrIdx < instances.numAttributes(); attrIdx++) {
				if (attrIdx == instances.classIndex())
					continue;
				if (instances.get(instIdx).value(attrIdx) > cutoffsForHigherValuesOfAttribute[attrIdx])
					K[instIdx]++;
			}
		}
		
		double cutoffOfKForTopClusters = Utils.getCutoffForACL(instances, K);
		
		for (int instIdx = 0; instIdx < instances.numInstances(); instIdx++) {
			
			if (K[instIdx] > cutoffOfKForTopClusters)
				instancesByCLA.instance(instIdx).setClassValue(positiveLabel);
			else
				instancesByCLA.instance(instIdx).setClassValue(Utils.getNegLabel(instancesByCLA, positiveLabel));
		}
		
		return instancesByCLA;
	}
	
	/**
	 * Calculate the final result and print the prediction result performance 
	 * in terms of TP, TN, FP, FN, precision, recall, and f1.
	 * @param instances
	 * @param experimental; boolean value whether experimental or not 
	 * @param filePath; string value of file name  
	 * @param supress detailed prediction results
	 * @param positiveLabel; string value of positive label 
	 */
	public void printResult(Instances instances, boolean experimental, String filePath, boolean suppress,
			String positiveLabel) {

		int TP = 0, FP = 0, TN = 0, FN = 0;

		for (int instIdx = 0; instIdx < instancesByCLA.numInstances(); instIdx++) {
			if (!suppress)
				System.out.println("CLA: Instance " + (instIdx + 1) + " predicted as, "
						+ Utils.getStringValueOfInstanceLabel(instancesByCLA, instIdx) + ", (Actual class: "
						+ Utils.getStringValueOfInstanceLabel(instances, instIdx) + ") ");

			if (!Double.isNaN(instances.get(instIdx).classValue())) {
				if (Utils.getStringValueOfInstanceLabel(instancesByCLA, instIdx)
						.equals(Utils.getStringValueOfInstanceLabel(instances, instIdx))) {
					if (Utils.getStringValueOfInstanceLabel(instancesByCLA, instIdx).equals(positiveLabel))
						TP++;
					else
						TN++;
				} else {
					if (Utils.getStringValueOfInstanceLabel(instancesByCLA, instIdx).equals(positiveLabel))
						FP++;
					else
						FN++;
				}
			}
		}

		if (TP + TN + FP + FN > 0)
			try {
				Utils.printEvaluationResultCLA(TP, TN, FP, FN, experimental, filePath);
			} catch (Exception e) {
				e.printStackTrace();
			}
		else if (suppress)
			System.out.println(
					"No labeled instances in the arff file. To see detailed prediction results, try again without the suppress option  (-s,--suppress)");

	}
}
