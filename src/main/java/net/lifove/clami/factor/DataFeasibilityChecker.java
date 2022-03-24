package net.lifove.clami.factor;

import java.util.ArrayList;
import java.util.Arrays;

import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;

import net.lifove.clami.CLA;
import weka.core.Instances;

public class DataFeasibilityChecker {

	private DataFactor numberOfInstance;
	private DataFactor numberOfMetrics;
	private DataFactor numberOfGroups;
	private DataFactor numberOfMaxVote;
	private DataFactor MaxValueOfVotes;	
	
	private ArrayList<DataFactor> factors = new ArrayList<DataFactor>();
	
	public void computeNumberOfGroups(Instances instancesByCLA, Instances instances, double percentileCutoff, String positiveLabel) {
		CLA cla = new CLA();
		
		int[] scoreOfInstances =  new int[instances.numInstances()];
		int numOfGroup = 0;
		int valueOfMaxVote = 0;
		int numOfMaxVote = 0;
		
		double alpha = 1.224;
		double criticalValue = alpha * Math.sqrt((instances.numInstances() * 2) / Math.pow(instances.numInstances(), 2));
	
		for (int attrIdx = 0; attrIdx < instances.numAttributes() - 1; attrIdx++) {

			// String metricsInTheGroup="";
			int k = 0;
			int flag=0;
			double[] X = instances.attributeToDoubleArray(attrIdx);
			Instances newInstances = new Instances(instances);

			for (int attrIdx1 = 0; attrIdx1 < instances.numAttributes() - 1; attrIdx1++) {
				
				double[] Y = instances.attributeToDoubleArray(attrIdx1);
				KolmogorovSmirnovTest ks = new KolmogorovSmirnovTest();
				double i = ks.kolmogorovSmirnovTest(X, Y);

				if (i < criticalValue) {
					newInstances.deleteAttributeAt(k);
				}
				if (i > criticalValue) {
					//check std of metric which is related to the main metric
					StandardDeviation std = new StandardDeviation();
					double stdOfMetric = std.evaluate(instances.attributeToDoubleArray(attrIdx1));
					if(stdOfMetric == 0.0) 
						flag=1;
					// metricsInTheGroup = metricsInTheGroup + Integer.toString(attrIdx1+1) + ",";
					k++;
				}
				
			}
			if(flag == 1)
				continue;

			if (newInstances.numAttributes() > 2) {
				instancesByCLA = cla.clustering(newInstances, percentileCutoff, positiveLabel);
				numOfGroup++;

				for (int instIdx = 0; instIdx < newInstances.numInstances(); instIdx++) {

					if (instancesByCLA.get(instIdx).value(newInstances.numAttributes() - 1) == Double
							.parseDouble(positiveLabel)) {
						scoreOfInstances[instIdx]++;
					}
				}
			}
			
			
		}
		
		Arrays.sort(scoreOfInstances);
		valueOfMaxVote = scoreOfInstances[instances.numInstances()-1] ;
		
		for (int instIdx = 0; instIdx < instances.numInstances(); instIdx++) {

			if (valueOfMaxVote == scoreOfInstances[instIdx]) {
				numOfMaxVote++;
			}
			
		}
		
		numberOfInstance = new DataFactor("numberOfInstances", instances.numInstances());
		numberOfMetrics = new DataFactor("numberOfMetrics", instances.numAttributes());
		numberOfGroups = new DataFactor("numberOfGroups", numOfGroup);
		numberOfMaxVote = new DataFactor("numberOfMaxVotes", numOfMaxVote);
		MaxValueOfVotes = new DataFactor("MaxValueOfVotes", valueOfMaxVote);
		
		addBasicFactors();
		addAdvancedFactor();
		
		printFactors();
	}

	private void addBasicFactors() {
		addDataFactor(numberOfInstance);
		addDataFactor(numberOfMetrics);
		addDataFactor(numberOfGroups);
		addDataFactor(numberOfMaxVote);
		addDataFactor(MaxValueOfVotes);
	}
	
	public void addAdvancedFactor() {
		DataFactorGIR gir = new DataFactorGIR(factors);
		DataFactorGMR gmr = new DataFactorGMR(factors);
		addDataFactor(gir.computeValue());
		addDataFactor(gmr.computeValue());
		
	}
	
	public void addDataFactor(DataFactor factor) {
		factors.add(factor);
	}
	
	private void printFactors() {
		for (int i = 0; i < factors.size(); i++) {
			System.out.println(factors.get(i).factorName + ": " + factors.get(i).getValue());
		}
	}
	
	public DataFactor getFactors(String name)
	{
		for(int i = 0; i < factors.size(); i++)
		{
			if(factors.get(i).factorName.equals(name))
				return factors.get(i);
		}		
		return null;
	}


}