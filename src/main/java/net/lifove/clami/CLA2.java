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

public class CLA2 extends CLA implements ICLA{
	double sum = 0.0;
	double totalViolation =0.0;
	double score =0.0;
	double[] instancesClassvalue;
	double[] instancesValue;
	Double[] K;
	HashMap<Double, Integer> percentileCorrelation;
	int j=1;
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
			

			
		for(j=1; j<100; j++)
		{
			
			instancesByCLA = clustering(instances, percentileCutoff, positiveLabel);
			printResult(instances, experimental, fileName, suppress, positiveLabel);
			
		}	
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


}
