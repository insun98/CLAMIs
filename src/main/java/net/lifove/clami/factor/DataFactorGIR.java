package net.lifove.clami.factor;

import java.util.Arrays;
import java.util.Collections;

import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;

import net.lifove.clami.CLA;
import weka.core.Instances;

public class DataFactorGIR extends DataFactor {

	public DataFactorGIR(Instances instancesByCLA, Instances instances, double percentileCutoff, String positiveLabel) {

		factorName = "GIR";
		computeValue(instancesByCLA, instances, percentileCutoff, positiveLabel);
		
	}
	
	/**
	 * This method calculates the data feature value which is GIR (# of groups / # of instances)
	 * @param instancesByCLA
	 * @param instances
	 * @param percentileCutoff
	 * @param positiveLabe
	 */
	@Override
	public void computeValue(Instances instancesByCLA, Instances instances, double percentileCutoff, String positiveLabel) {
		
		CLA cla = new CLA();
		
		int[] scoreOfInstances =  new int[instances.numInstances()];
		int numOfGroups = 0;
		int numOfInstances = instances.numInstances();
		
		double alpha = 1.224;
		double criticalValue = alpha * Math.sqrt((instances.numInstances() * 2) / Math.pow(instances.numInstances(), 2));
	//	System.out.println(criticalValue);
		int countOfSkip =0 ;
		for (int attrIdx = 0; attrIdx < instances.numAttributes() - 1; attrIdx++) {
			String metricsInTheGroup="";
			int k = 0;
			int flag = 0;
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
						flag = 1;
					metricsInTheGroup = metricsInTheGroup + Integer.toString(attrIdx1+1) + ",";
					k++;
				}
				
			}
			//skip the group if any metric in the group has std of 0
			if(flag == 1) {
				countOfSkip ++;
				continue;
			}
			//System.out.println(metricsInTheGroup);
			if (newInstances.numAttributes() > 2) {
				instancesByCLA = cla.clustering(newInstances, percentileCutoff, positiveLabel);
				numOfGroups++;

				for (int instIdx = 0; instIdx < newInstances.numInstances(); instIdx++) {

					if (instancesByCLA.get(instIdx).value(newInstances.numAttributes() - 1) == Double
							.parseDouble(positiveLabel)) {
						scoreOfInstances[instIdx]++;
					}
				}
			}
			
		}
		System.out.print("," + countOfSkip);
		System.out.print("," + numOfGroups);
		Arrays.sort(scoreOfInstances);
		System.out.println("," + scoreOfInstances[scoreOfInstances.length-1]);
		value = (double) numOfGroups/numOfInstances;
		
	}

	/**
	 * This method returns data feature value which is GIR. 
	 */
	@Override
	public double getValue() {
		return value;
	}

}
