package net.lifove.clami;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import org.apache.commons.math3.stat.correlation.SpearmansCorrelation;


import net.lifove.clami.percentileselectors.IPercentileSelector;
import net.lifove.clami.percentileselectors.PercentileSelectorBottom;
import net.lifove.clami.percentileselectors.PercentileSelectorTop;
import net.lifove.clami.util.Utils;
import weka.core.Instances;

public class CLA2 extends CLA implements ICLA, IPercentileSelector{
	double sum = 0.0;
	double totalViolation =0.0;
	double score =0.0;
	double[] instancesClassvalue;
	double[] instancesValue;
	Double[] K;
	HashMap<Double, Integer> percentileCorrelation;
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
		
	///	sum = 0.0;
	//	percentileCutoff = selectPercentileCutoff(instances, positiveLabel);
	//	for(percentileCutoff = 10.0; percentileCutoff<100; percentileCutoff+=5) {
	//		System.out.println("Percentile" + percentileCutoff);
			
			instancesByCLA = clustering(instances, percentileCutoff, positiveLabel);
			printResult(instances, experimental, fileName, suppress, positiveLabel);
		}
	//}
	
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
		K = new Double[instances.numInstances()];
		
		
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
		instancesClassvalue = new double[instances.numInstances()];
		
		for (int instIdx = 0; instIdx < instances.numInstances(); instIdx++) {
			
			if (K[instIdx] > cutoffOfKForTopClusters)
				instancesByCLA.instance(instIdx).setClassValue(positiveLabel);
			else
				instancesByCLA.instance(instIdx).setClassValue(Utils.getNegLabel(instancesByCLA, positiveLabel));
			
			instancesClassvalue[instIdx]=Double.parseDouble(Utils.getStringValueOfInstanceLabel(instancesByCLA, instIdx));
			//System.out.println("instancesClassvalue" + instancesClassvalue[instIdx]);
		}
		
		return instancesByCLA;
	}

	
	public double getOptimalPercentile(Instances instances, String positiveLabel, String percentileOption){
		
		double percentileCutoff;
		percentileCorrelation = new HashMap<>();
		int numOfCorrelation = 0;
		
		for(percentileCutoff = 10.0; percentileCutoff<100; percentileCutoff+=5){
			
			instancesByCLA = clustering(instances, percentileCutoff, positiveLabel);
			//System.out.println("Percentile" + percentileCutoff);
			
			
			for (int attrIdx = 0; attrIdx < instances.numAttributes(); attrIdx++) {
				
				instancesValue = instancesByCLA.attributeToDoubleArray(attrIdx);
				//System.out.println("instancesValue" + instancesValue[attrIdx]);
				 
				for(int i =0; i<instancesValue.length; i++) {
					if(Double.isNaN(instancesValue[i])) {
						instancesValue[i]=0;
					}
				}
				
				SpearmansCorrelation correlation1 = new SpearmansCorrelation();
					
				double correlation = correlation1.correlation(instancesValue, instancesClassvalue);
				if(Double.isNaN(correlation))
					correlation = 0;
						
				//System.out.println("metric num: " + (attrIdx+1) + " "+ correlation);
					
				if(correlation > 0.5)
					numOfCorrelation++;
			}
			
			percentileCorrelation.put(percentileCutoff, numOfCorrelation);
			numOfCorrelation = 0;
		}
		
		if(percentileOption.equals("t"))
			percentileCutoff = PercentileSelectorTop.getTopPercentileCutoff(instances,positiveLabel, percentileCorrelation);
		
		else if(percentileOption.equals("b"))
			percentileCutoff = PercentileSelectorBottom.getBottomPercentileCutoff(instances,positiveLabel,percentileCorrelation);
		
		else if(percentileOption.equals("m"))
			percentileCutoff = 50;
		
		
		
		return percentileCutoff;
		
	}

}
