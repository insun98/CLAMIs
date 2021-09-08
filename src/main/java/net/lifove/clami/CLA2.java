package net.lifove.clami;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.stream.*;

import org.apache.commons.math3.stat.correlation.SpearmansCorrelation;

import net.lifove.clami.util.Utils;
import weka.core.Instances;

public class CLA2 extends CLA implements ICLA{
	double sum = 0.0;
	double totalViolation =0.0;
	double score =0.0;
	/**
	 * Get CLA result
	 * @param instances
	 * @param percentileCutoff; cutoff percentile for cluster
	 * @param positiveLabel; string value of positive label 
	 * @param supress detailed prediction results
	 * @param fileName; string value of file name  
	 */
	public void getResult(Instances instances, double percentileCutoff, String positiveLabel, boolean suppress, String fileName) {
		getResult(instances, percentileCutoff, positiveLabel, suppress, false, fileName);
	}

	/**
	 * Get CLA result
	 * @param instances
	 * @param percentileCutoff; cutoff percentile for cluster
	 * @param positiveLabel; string value of positive label 
	 * @param supress detailed prediction results
	 * @param experimental; boolean value whether experimental or not 
	 * @param fileName; string value of file name  
	 */
	public void getResult(Instances instances, double percentileCutoff, String positiveLabel, boolean suppress,
			boolean experimental, String fileName) {
		
		sum = 0.0;
	//	percentileCutoff = selectPercentileCutoff(instances, positiveLabel);
		for(percentileCutoff = 10.0; percentileCutoff<100; percentileCutoff+=5) {
			System.out.println("Percentile" + percentileCutoff);
			instancesByCLA = clustering(instances, percentileCutoff, positiveLabel);
			printResult(instances, experimental, fileName, suppress, positiveLabel);
		}
	}
	
	/**
	 * Select the appropriate cutoff value
	 * @param instances
	 * @param positiveLabel
	 * @return
	 */
	public double selectPercentileCutoff(Instances instances, String positiveLabel) {
		double finalPercentileCutoff = 0.0;
		double percentileCutoff = 0.0;
		double maxScore =0.0;
		
		for(percentileCutoff = 10.0; percentileCutoff<100; percentileCutoff+=5) {
			maxScore = score;
			sum = 0.0;
			totalViolation =0.0;
			double[] cutoffsForHigherValuesOfAttribute = Utils.getHigherValueCutoffs(instances, percentileCutoff);
			
			instancesByCLA = clustering(instances, percentileCutoff, positiveLabel);
			
			HashMap<Integer, String>metricIdxWithTheSameViolationScores = Utils.getMetricIndicesWithTheViolationScores(instancesByCLA,
					cutoffsForHigherValuesOfAttribute, positiveLabel);
			Object[] keys = metricIdxWithTheSameViolationScores.keySet().toArray();
			
			for(Object key: keys){
				String key1 = key.toString();
				totalViolation = totalViolation + Integer.parseInt(key1);
			}
			
			score = (sum/instances.numAttributes()) + (totalViolation/(instances.numAttributes()*instances.numInstances()));
			System.out.println("percentile: " + percentileCutoff + " score: " + score +" violation"+ totalViolation);
			if(score < maxScore) 
				finalPercentileCutoff = percentileCutoff;
			}
		percentileCutoff = finalPercentileCutoff;
		return percentileCutoff;
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
				//System.out.println("median" + cutoffsForHigherValuesOfAttribute[attrIdx]);
				if (instances.get(instIdx).value(attrIdx) > cutoffsForHigherValuesOfAttribute[attrIdx]) 
					K[instIdx]++;
					
				
			}
		}
		double cutoffOfKForTopClusters = Utils.getMedian(new ArrayList<Double>(new HashSet<Double>(Arrays.asList(K))));
		double[] Y = new double[instances.numInstances()];
//		double []Y = {9, 7, 99, 6, 3, 0, 8, 0};
		for (int instIdx = 0; instIdx < instances.numInstances(); instIdx++) {
			
			if (K[instIdx] > cutoffOfKForTopClusters)
				instancesByCLA.instance(instIdx).setClassValue(positiveLabel);
			else
				instancesByCLA.instance(instIdx).setClassValue(Utils.getNegLabel(instancesByCLA, positiveLabel));
			
			Y[instIdx]=Double.parseDouble(Utils.getStringValueOfInstanceLabel(instancesByCLA, instIdx));
		}
		
		for (int attrIdx = 0; attrIdx < instances.numAttributes(); attrIdx++) {
			double[] X = instancesByCLA.attributeToDoubleArray(attrIdx);
		
			for(int i =0; i<X.length; i++) {
				if(Double.isNaN(X[i])) {
					X[i]=0;
				}
			}
			SpearmansCorrelation correlation1 = new SpearmansCorrelation();
			
			double correlation = correlation1.correlation(X,Y);
			if(Double.isNaN(correlation))
				correlation = 0;
			
			System.out.println("Metric No.: " + attrIdx + "correlation: " + correlation);
			sum = sum + correlation;

		}
		return instancesByCLA;
	}

	

}
