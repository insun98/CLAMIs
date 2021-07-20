package net.lifove.clami;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;

import net.lifove.clami.util.Utils;
import weka.core.Instances;

public class CLA implements ICLA {
	private Instances instancesByCLA = null;

	/**
	 * Get CLA result
	 * @param instances
	 * @param percentileCutoff; cutoff percentile for cluster
	 * @param positiveLabel; string value of positive label 
	 * @param supress detailed prediction results
	 * @param isDegree; boolean value whether plus version or not 
	 * @param fileName; string value of file name  
	 */
	public void getResult(Instances instances, double percentileCutoff, String positiveLabel, boolean suppress,
			boolean isDegree, String fileName) {
		getResult(instances, percentileCutoff, positiveLabel, suppress, false, isDegree, fileName);
	}

	/**
	 * Get CLA result
	 * @param instances
	 * @param percentileCutoff; cutoff percentile for cluster
	 * @param positiveLabel; string value of positive label 
	 * @param supress detailed prediction results
	 * @param experimental; boolean value whether experimental or not 
	 * @param isDegree; boolean value whether plus version or not 
	 * @param fileName; string value of file name  
	 */
	public void getResult(Instances instances, double percentileCutoff, String positiveLabel, boolean suppress,
			boolean experimental, boolean isDegree, String fileName) {

		if (isDegree)
			instancesByCLA = clusteringForContinuousValue(instances, percentileCutoff, positiveLabel);
		else
			instancesByCLA = clustering(instances, percentileCutoff, positiveLabel);
		printResult(instances, experimental, fileName, suppress, positiveLabel);
	}

	/**
	 * Cluster with percentileCutoff. Set class value to positive if K is higher than cutoff of cluster.
	 * @param instances
	 * @param percentileCutoff; cutoff percentile for cluster 
	 * @param positiveLabel; string value of positive label 
	 */
	public Instances clustering(Instances instances, double percentileCutoff, String positiveLabel) {

		instancesByCLA = new Instances(instances);
		double[] cutoffsForHigherValuesOfAttribute = Utils.getHigherValueCutoffs(instances, percentileCutoff);
		Double[] K = new Double[instances.numInstances()];

		for (int instIdx = 0; instIdx < instances.numInstances(); instIdx++) {
			K[instIdx] = 0.0;

			for (int attrIdx = 0; attrIdx < instances.numAttributes(); attrIdx++) {
				if (attrIdx == instances.classIndex())
					continue;
				if (instances.get(instIdx).value(attrIdx) > cutoffsForHigherValuesOfAttribute[attrIdx])
					K[instIdx]++;
			}
		}
		double cutoffOfKForTopClusters = Utils.getMedian(new ArrayList<Double>(new HashSet<Double>(Arrays.asList(K))));

		for (int instIdx = 0; instIdx < instances.numInstances(); instIdx++) {
			if (K[instIdx] > cutoffOfKForTopClusters)
				instancesByCLA.instance(instIdx).setClassValue(positiveLabel);
			else
				instancesByCLA.instance(instIdx).setClassValue(Utils.getNegLabel(instancesByCLA, positiveLabel));
		}

		return instancesByCLA;
	}

	/**
	 * Cluster with percentileCutoff. Set class value to positive if K is higher than cutoff of cluster. 
	 * This is for continuous value which is 'plus' version. 
	 * @param instances
	 * @param percentileCutoff; cutoff percentile for cluster 
	 * @param positiveLabel; string value of positive label 
	 */
	public Instances clusteringForContinuousValue(Instances instances, double percentileCutoff, String positiveLabel) {

		Instances instancesByCLA = new Instances(instances); 
		double[] cutoffsForHigherValuesOfAttribute = Utils.getHigherValueCutoffs(instances, percentileCutoff); 
		Double[] K = new Double[instances.numInstances()]; 

		for (int instIdx = 0; instIdx < instances.numInstances(); instIdx++) {
			K[instIdx] = 0.0;
			Double sum = 0.0;
			for (int attrIdx = 0; attrIdx < instances.numAttributes(); attrIdx++) {
				if (attrIdx == instances.classIndex())
					continue;
				sum = sum + 1 / (1 + Math.pow(Math.E,
						-(instances.get(instIdx).value(attrIdx) - cutoffsForHigherValuesOfAttribute[attrIdx])));

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

	/**
	 * Calculate the final result and print the prediction result performance 
	 * in terms of TP, TN, FP, FN, precision, recall, and f1.
	 * @param instances
	 * @param experimental; boolean value whether experimental or not 
	 * @param fileName; string value of file name  
	 * @param supress detailed prediction results
	 * @param positiveLabel; string value of positive label 
	 */
	public void printResult(Instances instances, boolean experimental, String fileName, boolean suppress,
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
				Utils.printEvaluationResultCLA(TP, TN, FP, FN, experimental, fileName);
			} catch (Exception e) {
				e.printStackTrace();
			}
		else if (suppress)
			System.out.println(
					"No labeled instances in the arff file. To see detailed prediction results, try again without the suppress option  (-s,--suppress)");

	}
}
