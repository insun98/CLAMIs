package net.lifove.clami;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;

import net.lifove.clami.util.Utils;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class Tools {
	CLA cla = new CLA();
	public double selectPercentileCutoff(Instances instances, String positiveLabel, Instances instancesByCLA, double score, double totalViolation, double sum) {
		double finalPercentileCutoff = 0.0;
		double percentileCutoff = 0.0;
		double maxScore =0.0;
		
		for(percentileCutoff = 10.0; percentileCutoff<100; percentileCutoff+=5) {
			maxScore = score;
			sum = 0.0;
			totalViolation =0.0;
			double[] cutoffsForHigherValuesOfAttribute = Utils.getHigherValueCutoffs(instances, percentileCutoff);
			
			instancesByCLA = cla.clustering(instances, percentileCutoff, positiveLabel);
			
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
	public void ksTest(Instances instances, double percentileCutoff, String positiveLabel, Instances instancesByCLA) {

		double criticalValue;
		double alpha = 1.224;
		int[] scoreOfInstances = new int[instances.numInstances()];
		int[] scoreOfInstancesCopy = new int[instances.numInstances()];
		
		ArrayList<ArrayList<Integer>> allIndex = new ArrayList<ArrayList<Integer>>() ;
		ArrayList<ArrayList<Double>> allPvalue = new ArrayList<ArrayList<Double>>() ;
		String[] idxArrToDelete = new String[instances.numAttributes()];
		
		Instances[] newnewInstances = new Instances[instances.numAttributes()];

		criticalValue = alpha * Math.sqrt((instances.numInstances()*2) / Math.pow(instances.numInstances(),2));
		System.out.println(criticalValue);
		//System.out.println(instances.numInstances());// 453

		for (int attrIdx = 0; attrIdx < instances.numAttributes()-1 ; attrIdx++) {

			ArrayList<Integer> attrIndex = new ArrayList<Integer>();
			ArrayList<Double> attrPvalue = new ArrayList<Double>();
			String indexToDelete = "";
			
			double[] X = instances.attributeToDoubleArray(attrIdx);
			newnewInstances[attrIdx] = new Instances(instances);

			for (int attrIdx1 = 0; attrIdx1 < instances.numAttributes()-1; attrIdx1++) {

				double[] Y = instances.attributeToDoubleArray(attrIdx1);
				// Instances newInstances = new Instances(instances);
				KolmogorovSmirnovTest ks = new KolmogorovSmirnovTest();
				double i = ks.kolmogorovSmirnovTest(X,Y);

				if ((attrIdx1+1) == instances.classIndex()) {
					continue;
				}
				
				if(i < criticalValue) { 
					indexToDelete += ((attrIdx1+1) + ","); // collect attribute index to remove
				}

				if(i > criticalValue) {
					attrIndex.add(attrIdx1);
					attrPvalue.add(i);
				}

			}
			idxArrToDelete[attrIdx] = indexToDelete;
			allIndex.add(attrIndex);
			allPvalue.add(attrPvalue);
			
//			System.out.print(attrIdx + " ["); 
			for (int i = 0; i < attrIndex.size();i++) {				
				System.out.print(attrIndex.get(i) + ", ");
			}
//			System.out.print("]\n");
			System.out.println("");
			
		}
		

		for (int attrIdx = 0; attrIdx < instances.numAttributes()-1 ; attrIdx++) {
			int size= 0;
			int maxAttr = 0 ;
			double maxPvalue = 0.0 ; 
			
			// find attribute of max p-value
			for (int j = 0; j < allIndex.get(attrIdx).size(); j++) {
				if (allPvalue.get(attrIdx).get(j) == 1) continue;
				if (maxPvalue < allPvalue.get(attrIdx).get(j)) {
					maxPvalue = allPvalue.get(attrIdx).get(j);
					maxAttr = allIndex.get(attrIdx).get(j);
				}
			}

			for (int i = 0; i < allIndex.get(attrIdx).size(); i++) {
				if (attrIdx+1 == instances.classIndex()) continue;
				
				if (allIndex.get(attrIdx).get(i) != maxAttr) {
					idxArrToDelete[allIndex.get(attrIdx).get(i)] += ((attrIdx+1) +",");  // collect attribute index to remove
				}
			}
		}
		
		// remove attributes and run CLA for all attributes 
		for (int attrIdx = 0; attrIdx < instances.numAttributes()-1 ; attrIdx++) {

			Remove remove = new Remove();
			remove.setAttributeIndices(idxArrToDelete[attrIdx]); // remove attributes
			try {
				remove.setInputFormat(newnewInstances[attrIdx]);
				newnewInstances[attrIdx] = Filter.useFilter(newnewInstances[attrIdx], remove); // remove attribute 
			} catch (Exception e) {
				e.printStackTrace();
				System.exit(0);
			}
			
			if(newnewInstances[attrIdx].numAttributes() > 2) { 
				//System.out.println(attrIdx);
				instancesByCLA =cla.clustering(newnewInstances[attrIdx], percentileCutoff, positiveLabel);

				for (int instIdx = 0; instIdx < newnewInstances[attrIdx].numInstances(); instIdx++) {
					if(instancesByCLA.get(instIdx).value(newnewInstances[attrIdx].numAttributes()-1) == Double.parseDouble(positiveLabel)) 
						scoreOfInstances[instIdx]++; 
				}
			}
		}
		
		
		scoreOfInstancesCopy =scoreOfInstances.clone();

		Arrays.sort(scoreOfInstances);
		int idx = (int)(instances.numInstances() * 0.25); 
		int cutoffValue= scoreOfInstances[idx];

		System.out.println(cutoffValue);

		for (int instIdx = 0; instIdx < instances.numInstances(); instIdx++) {

			if (scoreOfInstancesCopy[instIdx] >= cutoffValue)
				instancesByCLA.instance(instIdx).setClassValue(positiveLabel);
			else
				instancesByCLA.instance(instIdx).setClassValue(Utils.getNegLabel(instancesByCLA, positiveLabel));

			//Y[instIdx] = Double.parseDouble(Utils.getStringValueOfInstanceLabel(instancesByCLA, instIdx)) ;
		}

		Arrays.sort(scoreOfInstancesCopy);
		
		System.out.println("Overall");
		for(int i=0; i<scoreOfInstances.length; i++)
			System.out.println(scoreOfInstancesCopy[i] + " ");



	}


}
