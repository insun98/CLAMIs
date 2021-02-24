package net.lifove.clami.util;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import org.apache.commons.math3.stat.StatUtils;

import com.google.common.primitives.Doubles;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemoveRange;

public class Utils {
	
	
	/**
	 * Get CLA result
	 * @param instances
	 * @param percentileCutoff: cutoff percentile for top and bottom clusters
	 * @param positiveLabel positive label string value
	 * @param suppress detailed prediction results
	 * @return instances labeled by CLA
	 */
	public static void getCLAResult(Instances instances,double percentileCutoff,String positiveLabel,boolean suppress) {
		getCLAResult(instances,percentileCutoff,positiveLabel,suppress,false); // no experimental as default
	}
	
	/**
	 * Get CLA result
	 * @param instances
	 * @param percentileCutoff cutoff percentile for top and bottom clusters
	 * @param positiveLabel positive label string value
	 * @param suppress detailed prediction results
	 * @param experimental option to display a result in a line;
	 * @return instances labeled by CLA
	 */
	public static void getCLAResult(Instances instances,double percentileCutoff,String positiveLabel,boolean suppress,boolean experimental) {
		Instances instancesByCLA = getInstancesByCLA(instances, percentileCutoff, positiveLabel);
		
		// Print CLA results
		int TP=0, FP=0,TN=0, FN=0;
		for(int instIdx = 0; instIdx < instancesByCLA.numInstances(); instIdx++){
			if(!suppress)
				System.out.println("CLA: Instance " + (instIdx+1) + " predicted as, " + Utils.getStringValueOfInstanceLabel(instancesByCLA,instIdx) +
						", (Actual class: " + Utils.getStringValueOfInstanceLabel(instances,instIdx) + ") ");
			
			// compute T/F/P/N for the original instances labeled.
			if(!Double.isNaN(instances.get(instIdx).classValue())){
				if(Utils.getStringValueOfInstanceLabel(instancesByCLA,instIdx).equals(Utils.getStringValueOfInstanceLabel(instances,instIdx))){
					if(Utils.getStringValueOfInstanceLabel(instancesByCLA,instIdx).equals(positiveLabel))
						TP++;
					else
						TN++;
				}else{
					if(Utils.getStringValueOfInstanceLabel(instancesByCLA,instIdx).equals(positiveLabel))
						FP++;
					else
						FN++;
				}
			}
		}
		
		if (TP+TN+FP+FN>0)
			printEvaluationResult(TP, TN, FP, FN, experimental);
		else if(suppress)
			System.out.println("No labeled instances in the arff file. To see detailed prediction results, try again without the suppress option  (-s,--suppress)");
	}

	/**
	 * Print prediction performance in terms of TP, TN, FP, FN, precision, recall, and f1.
	 * @param tP
	 * @param tN
	 * @param fP
	 * @param fN
	 */
	private static void printEvaluationResult(int tP, int tN, int fP, int fN, boolean experimental) {
		
		double precision = (double)tP/(tP+fP);
		double recall = (double)tP/(tP+fN);
		double f1 = (2*(precision*recall))/(precision+recall);
		
		if(!experimental){
			System.out.println("TP: " + tP);
			System.out.println("FP: " + fP);
			System.out.println("TN: " + tN);
			System.out.println("FN: " + fN);
			
			System.out.println("Precision: " + precision);
			System.out.println("Recall: " + recall);
			System.out.println("F1: " + f1);
		}else{
			System.out.print(precision + "," + recall + "," + f1);
		}
	}

	/**
	 * Get instances labeled by CLA
	 * @param instances
	 * @param percentileCutoff
	 * @param positiveLabel
	 * @return
	 */
	private static Instances getInstancesByCLA(Instances instances, double percentileCutoff, String positiveLabel) {
		
		//System.out.println("\nHigher value cutoff > P" + percentileCutoff );
		
		Instances instancesByCLA = new Instances(instances);
		
		double[] cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(instances, percentileCutoff);
		
		// compute, K = the number of metrics whose values are greater than median, for each instance
		Double[] K = new Double[instances.numInstances()];
		
		for(int instIdx = 0; instIdx < instances.numInstances();instIdx++){
			K[instIdx]=0.0;
			for(int attrIdx = 0; attrIdx < instances.numAttributes();attrIdx++){
				if (attrIdx == instances.classIndex())
					continue;
				
				if(instances.get(instIdx).value(attrIdx) > cutoffsForHigherValuesOfAttribute[attrIdx]){
					K[instIdx]++;
				}
			}
		}
		
		// compute cutoff for the top half and bottom half clusters
		double cutoffOfKForTopClusters = Utils.getMedian(new ArrayList<Double>(new HashSet<Double>(Arrays.asList(K))));
		
		for(int instIdx = 0; instIdx < instances.numInstances(); instIdx++){
			if(K[instIdx]>cutoffOfKForTopClusters)
				instancesByCLA.instance(instIdx).setClassValue(positiveLabel);
			else
				instancesByCLA.instance(instIdx).setClassValue(getNegLabel(instancesByCLA,positiveLabel));
		}
		return instancesByCLA;
	}
	
	/**
	 * final labeling for probability of class
	 * labeling for testset1
	 * @param instances
	 * @param positiveLabel 
	 * @param positiveLabel
	 * @return
	 */
	private static Instances getLabeling(Instances instances, List<Double> v1_predictioned_label, List<Double> v2_predictioned_label, List<Double> v1, List<Double> v2, String positiveLabel) {
		
		//System.out.println("\nHigher value cutoff > P" + percentileCutoff );
		
		Instances instancesByCLA = new Instances(instances);
		
		for(int instIdx = 0; instIdx < instances.numInstances(); instIdx++){
			if(v1_predictioned_label.get(instIdx) != v2_predictioned_label.get(instIdx)) {
				if(v1.get(instIdx) < v2.get(instIdx)) {
					if(Utils.getStringValueOfInstanceLabel(instancesByCLA,instIdx) == positiveLabel) {
						instancesByCLA.instance(instIdx).setClassValue(getNegLabel(instancesByCLA,positiveLabel));
					} else {
						instancesByCLA.instance(instIdx).setClassValue(positiveLabel);
					}
				}	
			}
		}
		return instancesByCLA;
	}


	/**
	 * Get higher value cutoffs for each attribute
	 * @param instances
	 * @param percentileCutoff
	 * @return
	 */
	private static double[] getHigherValueCutoffs(Instances instances, double percentileCutoff) {
		// compute median values for attributes
		double[] cutoffForHigherValuesOfAttribute = new double[instances.numAttributes()];

		for(int attrIdx=0; attrIdx < instances.numAttributes();attrIdx++){
			if (attrIdx == instances.classIndex())
				continue;
			cutoffForHigherValuesOfAttribute[attrIdx] = StatUtils.percentile(instances.attributeToDoubleArray(attrIdx),percentileCutoff);
		}
		return cutoffForHigherValuesOfAttribute;
	}
	
	/**
	 * Get CLAMI result. Since CLAMI is the later steps of CLA, to get instancesByCLA use getCLAResult.
	 * @param testInstances
	 * @param instancesByCLA
	 * @param positiveLabel
	 */
	public static void getCLAMIResult(Instances testInstances, Instances instances, String positiveLabel,double percentileCutoff,boolean suppress,String mlAlg) {
		getCLAMIResult(testInstances,instances,positiveLabel,percentileCutoff,suppress,false,mlAlg); //no experimental as default
	}
	
	/**
	 * Get CLAMI result. Since CLAMI is the later steps of CLA, to get instancesByCLA use getCLAResult.
	 * @param testInstances
	 * @param instancesByCLA
	 * @param positiveLabel
	 */
	@SuppressWarnings("null")
	public static void getCLAMIResult(Instances testInstances, Instances instances, String positiveLabel,double percentileCutoff, boolean suppress, boolean experimental, String mlAlg) {
		
		String mlAlgorithm = mlAlg!=null && !mlAlg.equals("")?mlAlg:"weka.classifiers.functions.Logistic";
		
		Instances instancesByCLA = getInstancesByCLA(instances, percentileCutoff, positiveLabel);
		
		// Compute medians
		double[] cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(instancesByCLA,percentileCutoff);
				
		// Metric selection
		
		// (1) get distinct violation scores ordered by ASC
		HashMap<Integer,String> metricIdxWithTheSameViolationScores = getMetricIndicesWithTheViolationScores(instancesByCLA,cutoffsForHigherValuesOfAttribute,positiveLabel);
		Object[] keys = metricIdxWithTheSameViolationScores.keySet().toArray();
		Object[] descending_keys = metricIdxWithTheSameViolationScores.keySet().toArray();
		
		Arrays.sort(keys);
		Arrays.sort(descending_keys, Collections.reverseOrder());
		
		Instances trainingInstancesByCLAMI = null;
		Instances inverse_trainingInstancesByCLAMI = null;
		Instances final_trainingInstancesByCLAMI = instancesByCLA;
		
		// (2) Generate instances for CLAMI. If there are no instances in the first round with the minimum violation scores,
		//     then use the next minimum violation score. (Keys are ordered violation scores)
		Instances newTestInstances = null;
		Instances inverse_newTestInstances = null;
		Instances final_newTestInstances = instancesByCLA;
		
		// Ascending key
		for(Object key: keys){
			
			String selectedMetricIndices = metricIdxWithTheSameViolationScores.get(key) + (instancesByCLA.classIndex() +1);
			System.out.println("\nselected: " + selectedMetricIndices + "\n");
			trainingInstancesByCLAMI = getInstancesByRemovingSpecificAttributes(instancesByCLA,selectedMetricIndices,true);
			newTestInstances = getInstancesByRemovingSpecificAttributes(testInstances,selectedMetricIndices,true);
					
//			for(int instIdx = 0; instIdx < trainingInstancesByCLAMI.numInstances(); instIdx++){
//				for(int attrIdx = 0; attrIdx < trainingInstancesByCLAMI.numAttributes(); attrIdx++){
//					System.out.print(trainingInstancesByCLAMI.get(instIdx).value(attrIdx) + ", ");
//				}
//				System.out.println();
//			}
			
			// Instance selection
			cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(trainingInstancesByCLAMI,percentileCutoff); // get higher value cutoffs from the metric-selected dataset
			
			String instIndicesNeedToRemove = getSelectedInstances(trainingInstancesByCLAMI,cutoffsForHigherValuesOfAttribute,positiveLabel);
			trainingInstancesByCLAMI = getInstancesByRemovingSpecificInstances(trainingInstancesByCLAMI,instIndicesNeedToRemove,false);
			
//			for(int instIdx = 0; instIdx < trainingInstancesByCLAMI.numInstances(); instIdx++){
//				for(int attrIdx = 0; attrIdx < trainingInstancesByCLAMI.numAttributes(); attrIdx++){
//					System.out.print(trainingInstancesByCLAMI.get(instIdx).value(attrIdx) + ", ");
//				}
//				System.out.println();
//			}
			
			if(trainingInstancesByCLAMI.numInstances() != 0)
				break;
		}
		
		int v1_TP=0, v1_FP=0, v1_TN=0, v1_FN=0;
		double[] prediction;
		List<Double> v1 = new ArrayList<Double>();
		List<Double> v1_predictedLabelIdx = new ArrayList<Double>();
		
		// check if there are no instances in any one of two classes.
		if(trainingInstancesByCLAMI.attributeStats(trainingInstancesByCLAMI.classIndex()).nominalCounts[0]!=0 &&
				trainingInstancesByCLAMI.attributeStats(trainingInstancesByCLAMI.classIndex()).nominalCounts[1]!=0){
		
			try {
				Classifier classifier = (Classifier) weka.core.Utils.forName(Classifier.class, mlAlgorithm, null);
				classifier.buildClassifier(trainingInstancesByCLAMI);
				
				// Print CLAMI results
				for(int instIdx = 0; instIdx < newTestInstances.numInstances(); instIdx++){
					double predictedLabelIdx = classifier.classifyInstance(newTestInstances.get(instIdx));
					v1_predictedLabelIdx.add(classifier.classifyInstance(newTestInstances.get(instIdx)));
					
					if(!suppress)
						System.out.println("CLAMI: Instance " + (instIdx+1) + " predicted as, " + 
							newTestInstances.classAttribute().value((int)predictedLabelIdx)	+
							//((newTestInstances.classAttribute().indexOfValue(positiveLabel))==predictedLabelIdx?"buggy":"clean") +
							", (Actual class: " + Utils.getStringValueOfInstanceLabel(newTestInstances,instIdx) + ") ");
					// compute T/F/P/N for the original instances labeled.
					
					prediction = classifier.distributionForInstance(newTestInstances.get(instIdx));
					
					double max = prediction[0];
//					System.out.println("max: " + max);
					for(int i = 0; i < prediction.length; i++){
						System.out.println("Probability of class " + newTestInstances.classAttribute().value(i) + " : " + Double.toString(prediction[i]));
						if(max < prediction[i]) max = prediction[i];
					}
					
					v1.add(max);
					
					
					if(!Double.isNaN(instances.get(instIdx).classValue())){
						if(predictedLabelIdx==instances.get(instIdx).classValue()){
							if(predictedLabelIdx==instances.attribute(instances.classIndex()).indexOfValue(positiveLabel))
								v1_TP++;
							else
								v1_TN++;
						}else{
							if(predictedLabelIdx==instances.attribute(instances.classIndex()).indexOfValue(positiveLabel))
								v1_FP++;
							else
								v1_FN++;
						}
					}
				}
				
				Evaluation eval = new Evaluation(trainingInstancesByCLAMI);
				eval.evaluateModel(classifier, newTestInstances);
				
//				System.out.println("newTestInstances");
//				for(int instIdx = 0; instIdx < newTestInstances.numInstances(); instIdx++){
//					for(int attrIdx = 0; attrIdx < newTestInstances.numAttributes(); attrIdx++){
//						System.out.print(newTestInstances.get(instIdx).value(attrIdx) + ", ");
//					}
//					System.out.println();
//				}
				
				if (v1_TP+v1_TN+v1_FP+v1_FN>0){
					printEvaluationResult(v1_TP, v1_TN, v1_FP, v1_FN, experimental);
					// print AUC value
					if(!experimental)
						System.out.println("AUC: " + eval.areaUnderROC(newTestInstances.classAttribute().indexOfValue(positiveLabel)));
					else
						System.out.print("," + eval.areaUnderROC(newTestInstances.classAttribute().indexOfValue(positiveLabel)));
				}
				else if(suppress)
					System.out.println("No labeled instances in the arff file. To see detailed prediction results, try again without the suppress option  (-s,--suppress)");
				
			} catch (Exception e) {
				System.err.println("Specify the correct Weka machine learing classifier with a fully qualified name. E.g., weka.classifiers.functions.Logistic");
				e.printStackTrace();
				System.exit(0);
			}
		}else{
			System.err.println("Dataset is not proper to build a CLAMI model! Dataset does not follow the assumption, i.e. the higher metric value, the more bug-prone.");
		}
		
		// Descending key
		for(Object descending_key: descending_keys){
			
			if (Integer.parseInt(descending_key.toString()) >= 0) { //Integer.parseInt(inverse_key.toString())
										
				String inverse_selectedMetricIndices = metricIdxWithTheSameViolationScores.get(descending_key) + (instancesByCLA.classIndex() +1);
				System.out.println("\nselected: " + inverse_selectedMetricIndices);	
				
				inverse_trainingInstancesByCLAMI = getInstancesByRemovingSpecificAttributes(instancesByCLA,inverse_selectedMetricIndices,true);
				inverse_newTestInstances =	getInstancesByRemovingSpecificAttributes(testInstances,inverse_selectedMetricIndices,true);
						
//				for(int instIdx = 0; instIdx < inverse_trainingInstancesByCLAMI.numInstances(); instIdx++){
//	 				for(int attrIdx = 0; attrIdx < inverse_trainingInstancesByCLAMI.numAttributes(); attrIdx++){
//						System.out.print(inverse_trainingInstancesByCLAMI.get(instIdx).value(attrIdx) + ", ");
//					}
//					System.out.println();
//				}		
						
				// Instance selection
				cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(inverse_trainingInstancesByCLAMI,percentileCutoff); // get higher value cutoffs from the metric-selected dataset
						
				String inverse_instIndicesNeedToRemove = getSelectedInstances(inverse_trainingInstancesByCLAMI,cutoffsForHigherValuesOfAttribute,positiveLabel);
				inverse_trainingInstancesByCLAMI = getInstancesByRemovingSpecificInstances(inverse_trainingInstancesByCLAMI,inverse_instIndicesNeedToRemove,false);
						
//				System.out.println("final");
//				for(int instIdx = 0; instIdx < inverse_trainingInstancesByCLAMI.numInstances(); instIdx++){
//					for(int attrIdx = 0; attrIdx < inverse_trainingInstancesByCLAMI.numAttributes(); attrIdx++){
//						System.out.print(inverse_trainingInstancesByCLAMI.get(instIdx).value(attrIdx) + ", ");
//					}
//					System.out.println();
//				}
						
						
				if(inverse_trainingInstancesByCLAMI.numInstances() != 0) {
					break;
				}
			}
						
		}
		
		int v2_TP=0, v2_FP=0, v2_TN=0, v2_FN=0;
		double[] inverse_prediction;
		List<Double> v2 = new ArrayList<Double>();
		List<Double> v2_inverse_predictedLabelIdx = new ArrayList<Double>();
		
		if(inverse_trainingInstancesByCLAMI != null) {
			// check if there are no instances in any one of two classes.
			if(inverse_trainingInstancesByCLAMI.attributeStats(inverse_trainingInstancesByCLAMI.classIndex()).nominalCounts[0]!=0 &&
					inverse_trainingInstancesByCLAMI.attributeStats(inverse_trainingInstancesByCLAMI.classIndex()).nominalCounts[1]!=0){
			
				try {
					Classifier inverse_classifier = (Classifier) weka.core.Utils.forName(Classifier.class, mlAlgorithm, null);
					inverse_classifier.buildClassifier(inverse_trainingInstancesByCLAMI);
					
					// Print CLAMI results
					
					System.out.println();
					
					for(int instIdx = 0; instIdx < inverse_newTestInstances.numInstances(); instIdx++){
						double inverse_predictedLabelIdx = inverse_classifier.classifyInstance(inverse_newTestInstances.get(instIdx));
						v2_inverse_predictedLabelIdx.add(inverse_predictedLabelIdx);
						
						if(!suppress) {
							System.out.println("CLAMI: Instance " + (instIdx+1) + " predicted as, " + 
									inverse_newTestInstances.classAttribute().value((int)inverse_predictedLabelIdx)	+
									//((newTestInstances.classAttribute().indexOfValue(positiveLabel))==predictedLabelIdx?"buggy":"clean") +
									", (Actual class: " + Utils.getStringValueOfInstanceLabel(inverse_newTestInstances,instIdx) + ") ");
						}
						
						inverse_prediction = inverse_classifier.distributionForInstance(inverse_newTestInstances.get(instIdx));
						
						double max = inverse_prediction[0];
						
						for(int i = 0; i < inverse_prediction.length; i++){
//							System.out.println("Probability of class " + inverse_newTestInstances.classAttribute().value(i) + " : " + Double.toString(inverse_prediction[i]));
							if(max < inverse_prediction[i]) max = inverse_prediction[i];
						}
						
//						System.out.println("max: " + max);
						v2.add(max);
						
						// compute T/F/P/N for the original instances labeled.
						if(!Double.isNaN(instances.get(instIdx).classValue())){
							
							if(inverse_predictedLabelIdx==instances.get(instIdx).classValue()){
								if(inverse_predictedLabelIdx==instances.attribute(instances.classIndex()).indexOfValue(positiveLabel))
									v2_TP++;
								else
									v2_TN++;
							}else{
								if(inverse_predictedLabelIdx==instances.attribute(instances.classIndex()).indexOfValue(positiveLabel))
									v2_FP++;
								else
									v2_FN++;
							}
						}
					}
//					System.out.println("inverse_trainingInstancesByCLAMI");
//					for(int instIdx = 0; instIdx < inverse_trainingInstancesByCLAMI.numInstances(); instIdx++){
//						for(int attrIdx = 0; attrIdx < inverse_trainingInstancesByCLAMI.numAttributes(); attrIdx++){
//							System.out.print(inverse_trainingInstancesByCLAMI.get(instIdx).value(attrIdx) + ", ");
//						}
//						System.out.println();
//					}
					
					Evaluation inverse_eval = new Evaluation(inverse_trainingInstancesByCLAMI);
					inverse_eval.evaluateModel(inverse_classifier, inverse_newTestInstances);
					
//					System.out.println("inverse_newTestInstances");
//					for(int instIdx = 0; instIdx < inverse_newTestInstances.numInstances(); instIdx++){
//						for(int attrIdx = 0; attrIdx < inverse_newTestInstances.numAttributes(); attrIdx++){
//							System.out.print(inverse_newTestInstances.get(instIdx).value(attrIdx) + ", ");
//						}
//						System.out.println();
//					}
					
					if (v2_TP+v2_TN+v2_FP+v2_FN>0){
						printEvaluationResult(v2_TP, v2_TN, v2_FP, v2_FN, experimental);
						System.out.println("TP + TN: " + (v2_TP + v2_TN));
						// print AUC value
						if(!experimental)
							System.out.println("AUC: " + inverse_eval.areaUnderROC(inverse_newTestInstances.classAttribute().indexOfValue(positiveLabel)));
						else
							System.out.print("," + inverse_eval.areaUnderROC(inverse_newTestInstances.classAttribute().indexOfValue(positiveLabel)));
					}
					else if(suppress)
						System.out.println("No labeled instances in the arff file. To see detailed prediction results, try again without the suppress option  (-s,--suppress)");
					
				} catch (Exception e) {
					System.err.println("Specify the correct Weka machine learing classifier with a fully qualified name. E.g., weka.classifiers.functions.Logistic");
					e.printStackTrace();
					System.exit(0);
				}
			}else{
				System.err.println("Dataset is not proper to build a CLAMI model! Dataset does not follow the assumption, i.e. the higher metric value, the more bug-prone.");
			}
		} else {
			System.out.println("Does not inverse case!!");
		}
		
//		System.out.println("before final");
//		for(int instIdx = 0; instIdx < final_newTestInstances.numInstances(); instIdx++){
//			for(int attrIdx = 0; attrIdx < final_newTestInstances.numAttributes(); attrIdx++){
//				System.out.print(final_newTestInstances.get(instIdx).value(attrIdx) + ", ");
//			}
//			System.out.println();
//		}
		
		Instances labeling = getLabeling(final_newTestInstances, v1_predictedLabelIdx, v2_inverse_predictedLabelIdx, v1, v2, positiveLabel); // CLA로 라벨링 된 set
		
		
//		System.out.println("after final");
//		for(int instIdx = 0; instIdx < labeling.numInstances(); instIdx++){
//			for(int attrIdx = 0; attrIdx < labeling.numAttributes(); attrIdx++){
//				System.out.print(labeling.get(instIdx).value(attrIdx) + ", ");
//			}
//			System.out.println();
//		}
		
		int TP=0, FP=0, TN=0, FN=0;
		double[] final_prediction;
		
		if(labeling != null) {
			// check if there are no instances in any one of two classes.
			if(labeling.attributeStats(labeling.classIndex()).nominalCounts[0]!=0 &&
					labeling.attributeStats(labeling.classIndex()).nominalCounts[1]!=0){
			
				try {
					Classifier final_classifier = (Classifier) weka.core.Utils.forName(Classifier.class, mlAlgorithm, null);
					final_classifier.buildClassifier(labeling);
					
					// Print CLAMI results
					
					System.out.println();
					
					for(int instIdx = 0; instIdx < final_newTestInstances.numInstances(); instIdx++){
//						double final_predictedLabelIdx = final_classifier.classifyInstance(final_newTestInstances.get(instIdx));
						double final_predictedLabelIdx = v1_predictedLabelIdx.get(instIdx);
						if (!(v1_predictedLabelIdx.get(instIdx).equals(v2_inverse_predictedLabelIdx.get(instIdx)))) {
							if (v1.get(instIdx) < v2.get(instIdx)) {
								final_predictedLabelIdx = v2_inverse_predictedLabelIdx.get(instIdx);
							}
						}
						
//						System.out.println("final predicted Label Index : " + final_predictedLabelIdx);
						
						
						if(!suppress) {
							System.out.println("CLAMI: Instance " + (instIdx+1) + " predicted as, " + 
									final_newTestInstances.classAttribute().value((int)final_predictedLabelIdx)	+
									//((newTestInstances.classAttribute().indexOfValue(positiveLabel))==predictedLabelIdx?"buggy":"clean") +
									", (Actual class: " + Utils.getStringValueOfInstanceLabel(final_newTestInstances,instIdx) + ") ");
						}
						
						final_prediction = final_classifier.distributionForInstance(final_newTestInstances.get(instIdx));
						
//						for(int i = 0; i < final_prediction.length; i++){
//							
//							System.out.println("Probability of class " + final_newTestInstances.classAttribute().value(i) + " : " + Double.toString(final_prediction[i]));
//						}
						
						
						
						// compute T/F/P/N for the original instances labeled.
						if(!Double.isNaN(instances.get(instIdx).classValue())){
							
							if(final_predictedLabelIdx==instances.get(instIdx).classValue()){
								if(final_predictedLabelIdx==instances.attribute(instances.classIndex()).indexOfValue(positiveLabel)) {
									TP++;
									System.out.println("TP");
								}
								else {
									TN++;
									System.out.println("TN");
								}
							}else{
								if(final_predictedLabelIdx==instances.attribute(instances.classIndex()).indexOfValue(positiveLabel)) {
									FP++;
									System.out.println("FP");
								}
								else {
									FN++;
									System.out.println("FN");
								}
							}
						}
					}
					
					Evaluation final_eval = new Evaluation(labeling);
					final_eval.evaluateModel(final_classifier, final_newTestInstances);
					
					if (TP+TN+FP+FN>0){
						printEvaluationResult(TP, TN, FP, FN, experimental);
						System.out.println("TP + TN: " + (TP + TN));
						// print AUC value
						if(!experimental)
							System.out.println("AUC: " + final_eval.areaUnderROC(final_newTestInstances.classAttribute().indexOfValue(positiveLabel)));
						else
							System.out.print("," + final_eval.areaUnderROC(final_newTestInstances.classAttribute().indexOfValue(positiveLabel)));
					}
					else if(suppress)
						System.out.println("No labeled instances in the arff file. To see detailed prediction results, try again without the suppress option  (-s,--suppress)");
					
				} catch (Exception e) {
					System.err.println("Specify the correct Weka machine learing classifier with a fully qualified name. E.g., weka.classifiers.functions.Logistic");
					e.printStackTrace();
					System.exit(0);
				}
			}else{
				System.err.println("Dataset is not proper to build a CLAMI model! Dataset does not follow the assumption, i.e. the higher metric value, the more bug-prone.");
			}
		} else {
			System.out.println("Does not inverse case!!");
		}
	}

	private static HashMap<Integer, String> getMetricIndicesWithTheViolationScores(Instances instances,
			double[] cutoffsForHigherValuesOfAttribute, String positiveLabel) {

		int[] violations = new int[instances.numAttributes()];
		
		for(int attrIdx=0; attrIdx < instances.numAttributes(); attrIdx++){
			if(attrIdx == instances.classIndex()){
				violations[attrIdx] = instances.numInstances(); // make this as max to ignore since our concern is minimum violation.
				continue;
			}
			
			for(int instIdx=0; instIdx < instances.numInstances(); instIdx++){
				if (instances.get(instIdx).value(attrIdx) <= cutoffsForHigherValuesOfAttribute[attrIdx]
						&& instances.get(instIdx).classValue() == instances.classAttribute().indexOfValue(positiveLabel)){
						violations[attrIdx]++;
				}else if(instances.get(instIdx).value(attrIdx) > cutoffsForHigherValuesOfAttribute[attrIdx]
						&& instances.get(instIdx).classValue() == instances.classAttribute().indexOfValue(getNegLabel(instances, positiveLabel))){
						violations[attrIdx]++;
				}
			}
		}
		
		HashMap<Integer,String> metricIndicesWithTheSameViolationScores = new HashMap<Integer,String>();
		
		for(int attrIdx=0; attrIdx < instances.numAttributes(); attrIdx++){
			if(attrIdx == instances.classIndex()){
				continue;
			}
			
			int key = violations[attrIdx];
			
			if(!metricIndicesWithTheSameViolationScores.containsKey(key)){
				metricIndicesWithTheSameViolationScores.put(key,(attrIdx+1) + ",");
			}else{
				String indices = metricIndicesWithTheSameViolationScores.get(key) + (attrIdx+1) + ",";
				metricIndicesWithTheSameViolationScores.put(key,indices);
			}
		}
		
		return metricIndicesWithTheSameViolationScores;
	}

	private static String getSelectedInstances(Instances instances, double[] cutoffsForHigherValuesOfAttribute,
			String positiveLabel) {
		
		int[] violations = new int[instances.numInstances()];
		
		for(int instIdx=0; instIdx < instances.numInstances(); instIdx++){
			
			for(int attrIdx=0; attrIdx < instances.numAttributes(); attrIdx++){
				if(attrIdx == instances.classIndex())
					continue; // no need to compute violation score for the class attribute
				
				if (instances.get(instIdx).value(attrIdx) <= cutoffsForHigherValuesOfAttribute[attrIdx]
						&& instances.get(instIdx).classValue() == instances.classAttribute().indexOfValue(positiveLabel)){
						violations[instIdx]++;
				}else if(instances.get(instIdx).value(attrIdx) > cutoffsForHigherValuesOfAttribute[attrIdx]
						&& instances.get(instIdx).classValue() == instances.classAttribute().indexOfValue(getNegLabel(instances, positiveLabel))){
						violations[instIdx]++;
				}
			}
		}
		
		String selectedInstances = "";
		
		for(int instIdx=0; instIdx < instances.numInstances(); instIdx++){
			if(violations[instIdx]>0)
				selectedInstances += (instIdx+1) + ","; // let the start attribute index be 1 
		}
		
		return selectedInstances;
	}
	
	/**
	 * Get the negative label string value from the positive label value
	 * @param instances
	 * @param positiveLabel
	 * @return
	 */
	static public String getNegLabel(Instances instances, String positiveLabel){
		if(instances.classAttribute().numValues()==2){
			int posIndex = instances.classAttribute().indexOfValue(positiveLabel);
			if(posIndex==0)
				return instances.classAttribute().value(1);
			else
				return instances.classAttribute().value(0);
		}
		else{
			System.err.println("Class labels must be binary");
			System.exit(0);
		}
		return null;
	}
	
	/**
	 * Load Instances from arff file. Last attribute will be set as class attribute
	 * @param path arff file path
	 * @return Instances
	 */
	public static Instances loadArff(String path,String classAttributeName){
		Instances instances=null;
		BufferedReader reader;
		try {
			reader = new BufferedReader(new FileReader(path));
			instances = new Instances(reader);
			reader.close();
			instances.setClassIndex(instances.attribute(classAttributeName).index());
		} catch (NullPointerException e) {
			System.err.println("Class label name, " + classAttributeName + ", does not exist! Please, check if the label name is correct.");
			instances = null;
		} catch (FileNotFoundException e) {
			System.err.println("Data file, " +path + ", does not exist. Please, check the path again!");
		} catch (IOException e) {
			System.err.println("I/O error! Please, try again!");
		}

		return instances;
	}
	
	/**
	 * Get label value of an instance
	 * @param instances
	 * @param instance index
	 * @return string label of an instance
	 */
	static public String getStringValueOfInstanceLabel(Instances instances,int intanceIndex){
		return instances.instance(intanceIndex).stringValue(instances.classIndex());
	}
	
	/**
	 * Get median from ArraList<Double>
	 * @param values
	 * @return
	 */
	static public double getMedian(ArrayList<Double> values){
		return getPercentile(values,50);
	}
	
	/**
	 * Get a value in a specific percentile from ArraList<Double>
	 * @param values
	 * @return
	 */
	static public double getPercentile(ArrayList<Double> values,double percentile){
		return StatUtils.percentile(getDoublePrimitive(values),percentile);
	}
	
	/**
	 * Get primitive double form ArrayList<Double>
	 * @param values
	 * @return
	 */
	public static double[] getDoublePrimitive(ArrayList<Double> values) {
		return Doubles.toArray(values);
	}
	
	/**
	 * Get instances by removing specific attributes
	 * @param instances
	 * @param attributeIndices attribute indices (e.g., 1,3,4) first index is 1
	 * @param invertSelection for invert selection, if true, select attributes with attributeIndices bug if false, remote attributes with attributeIndices
	 * @return new instances with specific attributes
	 */
	static public Instances getInstancesByRemovingSpecificAttributes(Instances instances,String attributeIndices,boolean invertSelection){
		Instances newInstances = new Instances(instances);

		Remove remove;

		remove = new Remove();
		remove.setAttributeIndices(attributeIndices);
		remove.setInvertSelection(invertSelection);
		try {
			remove.setInputFormat(newInstances);
			newInstances = Filter.useFilter(newInstances, remove);
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(0);
		}

		return newInstances;
	}
	
	/**
	 * Get instances by removing specific instances
	 * @param instances
	 * @param instance indices (e.g., 1,3,4) first index is 1
	 * @param option for invert selection
	 * @return selected instances
	 */
	static public Instances getInstancesByRemovingSpecificInstances(Instances instances,String instanceIndices,boolean invertSelection){
		Instances newInstances = null;

		RemoveRange instFilter = new RemoveRange();
		instFilter.setInstancesIndices(instanceIndices);
		instFilter.setInvertSelection(invertSelection);

		try {
			instFilter.setInputFormat(instances);
			newInstances = Filter.useFilter(instances, instFilter);
		} catch (Exception e) {
			e.printStackTrace();
		}

		return newInstances;
	}
}