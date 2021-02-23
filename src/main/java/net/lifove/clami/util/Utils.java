package net.lifove.clami.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
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
import weka.core.converters.ConverterUtils.DataSource;

public class Utils {
	
	static String fileName = new String();
	
	
	/**
	 * Get CLA result
	 * @param instances
	 * @param percentileCutoff: cutoff percentile for top and bottom clusters
	 * @param positiveLabel positive label string value
	 * @param suppress detailed prediction results
	 * @return instances labeled by CLA
	 */
	public static void getCLAResult(Instances instances,double percentileCutoff, double threshold, String positiveLabel,boolean suppress) {
		getCLAResult(instances,percentileCutoff, threshold, positiveLabel,suppress,false); // no experimental as default
	}
	
	/**
	 * Get CLA result
	 * @param instances
	 * @param percentileCutoff cutoff percentile for top and bottom clusters
	 * @param threshold 
	 * @param positiveLabel positive label string value
	 * @param suppress detailed prediction results
	 * @param experimental option to display a result in a line;
	 * @return instances labeled by CLA
	 */
	public static void getCLAResult(Instances instances,double percentileCutoff,double threshold, String positiveLabel,boolean suppress,boolean experimental) {
		Instances instancesByCLA = getInstancesByCLA(instances, percentileCutoff, threshold, positiveLabel);
		
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
			String[] array = fileName.split("/");
			fileName = array[array.length-1];
			
			System.out.print(fileName+","+tP + "," + fP + ","+tN+","+fN + ","+precision+","+recall+","+f1+",");
//			System.out.println("TP: " + tP);
//			System.out.println("FP: " + fP);
//			System.out.println("TN: " + tN);
//			System.out.println("FN: " + fN);
//			
//			System.out.println("Precision: " + precision);
//			System.out.println("Recall: " + recall);
//			System.out.println("F1: " + f1);
//			System.out.println(fileName);
			System.out.println();
		}else{
			System.out.print(precision + "," + recall + "," + f1);
		}
		
		writeCSV(tP, tN, fP, fN, precision, recall, f1, experimental);
	}

	private static void writeCSV(int tP, int tN, int fP, int fN, double precision, double recall, double f1, boolean experimental) {		
		
		//출력 스트림 생성
        BufferedWriter bufWriter = null;
        try{
            bufWriter = Files.newBufferedWriter(Paths.get("C:\\Users\\park_\\git\\CLAMI_BI\\CLAMI_v2\\clami_result.csv"),Charset.forName("UTF-8"));
            
            //csv파일 읽기
            List<List<String>> allData = readCSV();
            
            for(List<String> newLine : allData){
                List<String> list = newLine;
                for(String data : list){
                    bufWriter.write(data);
                    bufWriter.write(",");
                }
                //추가하기
//                bufWriter.write("주소"); // 줄 끝에 추가
                //개행코드추가
                bufWriter.newLine();
            }
          //추가하기
            if(!experimental){
            	bufWriter.write(fileName);
                bufWriter.write(",");
                bufWriter.write(String.valueOf(tP));
                bufWriter.write(",");
                bufWriter.write(String.valueOf(fP));
                bufWriter.write(",");
                bufWriter.write(String.valueOf(tN));
                bufWriter.write(",");
                bufWriter.write(String.valueOf(fN));
                bufWriter.write(",");
                bufWriter.write(String.valueOf(precision));
                bufWriter.write(",");
                bufWriter.write(String.valueOf(recall));
                bufWriter.write(",");
                bufWriter.write(String.valueOf(f1));
    		}else{
    			bufWriter.write(",");
    			bufWriter.write(",");
    			bufWriter.write(",");
    			bufWriter.write(",");
    			bufWriter.write(",");
    			bufWriter.write(String.valueOf(precision));
                bufWriter.write(",");
                bufWriter.write(String.valueOf(recall));
                bufWriter.write(",");
                bufWriter.write(String.valueOf(f1));
    		}
            
        }catch(FileNotFoundException e){
            e.printStackTrace();
        }catch(IOException e){
            e.printStackTrace();
        }finally{
            try{
                if(bufWriter != null){
                    bufWriter.close();
                }
            }catch(IOException e){
                e.printStackTrace();
            }
        }
    }
    
    public static List<List<String>> readCSV(){
        //반환용 리스트
        List<List<String>> ret = new ArrayList<List<String>>();
        BufferedReader br = null;
        File csv = new File("clami_result.csv");
        String line = "";
        
        try{
        	br = new BufferedReader(new FileReader(csv));
            //Charset.forName("UTF-8");
            
            while((line = br.readLine()) != null){
                //CSV 1행을 저장하는 리스트
            	System.out.println(line);
                List<String> tmpList = new ArrayList<String>();
                String array[] = line.split(",");
                //배열에서 리스트 반환
                tmpList = Arrays.asList(array);
                System.out.println(tmpList);
                ret.add(tmpList);
            }
        }catch(FileNotFoundException e){
            e.printStackTrace();
        }catch(IOException e){
            e.printStackTrace();
        }finally{
            try{
                if(br != null){
                    br.close();
                }
            }catch(IOException e){
                e.printStackTrace();
            }
        }
        return ret;
    }

	/**
	 * Get instances labeled by CLA
	 * @param instances
	 * @param percentileCutoff
	 * @param positiveLabel
	 * @return
	 */
	private static Instances getInstancesByCLA(Instances instances, double percentileCutoff, double threshold, String positiveLabel) {
		
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
	public static void getCLAMIResult(Instances testInstances, Instances instances, String positiveLabel,double percentileCutoff, double threshold, boolean suppress,String mlAlg) {
		getCLAMIResult(testInstances,instances,positiveLabel,percentileCutoff, threshold, suppress,false,mlAlg); //no experimental as default
	}
	
	/**
	 * Get CLAMI result. Since CLAMI is the later steps of CLA, to get instancesByCLA use getCLAResult.
	 * @param testInstances
	 * @param instancesByCLA
	 * @param positiveLabel
	 * @param threshold 
	 */
	public static void getCLAMIResult(Instances testInstances, Instances instances, String positiveLabel,double percentileCutoff, double threshold, boolean suppress, boolean experimental, String mlAlg) {
		
		String mlAlgorithm = mlAlg!=null && !mlAlg.equals("")?mlAlg:"weka.classifiers.functions.Logistic";
		
		Instances instancesByCLA = getInstancesByCLA(instances, percentileCutoff, threshold, positiveLabel);
		
		// Compute medians
		double[] cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(instancesByCLA,percentileCutoff);
				
		// Metric selection
		
		// (1) get distinct violation scores ordered by ASC
		HashMap<Integer,String> metricIdxWithTheSameViolationScores = getMetricIndicesWithTheViolationScores(instancesByCLA,cutoffsForHigherValuesOfAttribute,positiveLabel);
		Object[] keys = metricIdxWithTheSameViolationScores.keySet().toArray();
		Arrays.sort(keys);
		
		Instances trainingInstancesByCLAMI = null;
		
		// (2) Generate instances for CLAMI. If there are no instances in the first round with the minimum violation scores,
		//     then use the next minimum violation score. (Keys are ordered violation scores)
		Instances newTestInstances = null;
		for(Object key: keys){
			
			String selectedMetricIndices = metricIdxWithTheSameViolationScores.get(key) + (instancesByCLA.classIndex() +1);
			trainingInstancesByCLAMI = getInstancesByRemovingSpecificAttributes(instancesByCLA,selectedMetricIndices,true);
			newTestInstances = getInstancesByRemovingSpecificAttributes(testInstances,selectedMetricIndices,true);
					
			// Instance selection
			cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(trainingInstancesByCLAMI,percentileCutoff); // get higher value cutoffs from the metric-selected dataset
			String instIndicesNeedToRemove = getSelectedInstances(trainingInstancesByCLAMI,"","",cutoffsForHigherValuesOfAttribute,positiveLabel);
			trainingInstancesByCLAMI = getInstancesByRemovingSpecificInstances(trainingInstancesByCLAMI,instIndicesNeedToRemove,false);
			
			if(trainingInstancesByCLAMI.numInstances() != 0)
				break;
		}
		
		// check if there are no instances in any one of two classes.
		if(trainingInstancesByCLAMI.attributeStats(trainingInstancesByCLAMI.classIndex()).nominalCounts[0]!=0 &&
				trainingInstancesByCLAMI.attributeStats(trainingInstancesByCLAMI.classIndex()).nominalCounts[1]!=0){
		
			try {
				Classifier classifier = (Classifier) weka.core.Utils.forName(Classifier.class, mlAlgorithm, null);
				classifier.buildClassifier(trainingInstancesByCLAMI);
				
				// Print CLAMI results
				int TP=0, FP=0,TN=0, FN=0;
				for(int instIdx = 0; instIdx < newTestInstances.numInstances(); instIdx++){
					double[] probability = classifier.distributionForInstance(newTestInstances.get(instIdx));
					
//					for(int i = 0; i < probability.length; i++){
//						System.out.println("Probability of class " + newTestInstances.classAttribute().value(i) + " : " + Double.toString(probability[i]));
//					}
					
					double predictedLabelIdx = (probability[0] >= threshold)? 0.0: 1.0;
					
					if(!suppress)
						System.out.println("CLAMI: Instance " + (instIdx+1) + " predicted as, " + 
							newTestInstances.classAttribute().value((int)predictedLabelIdx)	+
							//((newTestInstances.classAttribute().indexOfValue(positiveLabel))==predictedLabelIdx?"buggy":"clean") +
							", (Actual class: " + Utils.getStringValueOfInstanceLabel(newTestInstances,instIdx) + ") ");

					
					// compute T/F/P/N for the original instances labeled.
					if(!Double.isNaN(instances.get(instIdx).classValue())){
						if(predictedLabelIdx==instances.get(instIdx).classValue()){
							if(predictedLabelIdx==instances.attribute(instances.classIndex()).indexOfValue(positiveLabel))
								TP++;
							else
								TN++;
						}else{
							if(predictedLabelIdx==instances.attribute(instances.classIndex()).indexOfValue(positiveLabel))
								FP++;
							else
								FN++;
						}
					}
				}
				
				Evaluation eval = new Evaluation(trainingInstancesByCLAMI);
				eval.evaluateModel(classifier, newTestInstances);
				
				if (TP+TN+FP+FN>0){
					printEvaluationResult(TP, TN, FP, FN, experimental);
					// print AUC value
//					if(!experimental)
//						System.out.println("AUC: " + eval.areaUnderROC(newTestInstances.classAttribute().indexOfValue(positiveLabel)));
//					else
//						System.out.print("," + eval.areaUnderROC(newTestInstances.classAttribute().indexOfValue(positiveLabel)));
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
	}
	
	// ---------------------------------------------------------------------------------------------------------------- //
	
	/**
	 * Get CLAMI result. Since CLAMI is the later steps of CLA, to get instancesByCLA use getCLAResult. WITH INVERSE 
	 * @param testInstances
	 * @param instancesByCLA
	 * @param positiveLabel
	 */
	public static void getCLAMIResult2(Instances testInstances, Instances instances, String positiveLabel,double percentileCutoff,double threshold,boolean suppress,String mlAlg, double inversePercent) {
		getCLAMIResult2(testInstances,instances,positiveLabel,percentileCutoff,threshold,suppress,false,mlAlg,inversePercent); //no experimental as default
	}
	
	/**
	 * Get CLAMI result. Since CLAMI is the later steps of CLA, to get instancesByCLA use getCLAResult. ** WITH INVERSE **
	 * @param testInstances
	 * @param instancesByCLA
	 * @param positiveLabel
	 * @param threshold 
	 */
	public static void getCLAMIResult2(Instances testInstances, Instances instances, String positiveLabel,double percentileCutoff, double threshold, boolean suppress, boolean experimental, String mlAlg, double inversePercent) {
		
		String mlAlgorithm = mlAlg!=null && !mlAlg.equals("")?mlAlg:"weka.classifiers.functions.Logistic";
		
		Instances instancesByCLA = getInstancesByCLA(instances, percentileCutoff, threshold, positiveLabel);
		
		// Compute medians
		double[] cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(instancesByCLA,percentileCutoff);
				
		// Metric selection
		
		// (1) get distinct violation scores ordered by ASC
		HashMap<Integer,String> metricIdxWithTheSameViolationScores = getMetricIndicesWithTheViolationScores(instancesByCLA,cutoffsForHigherValuesOfAttribute,positiveLabel);
		Object[] keys = metricIdxWithTheSameViolationScores.keySet().toArray();
		Arrays.sort(keys);
		
		Double minPer = (double) ((int) keys[0]/ (double)instancesByCLA.numInstances())*100.0;
		Double maxPer = (double) ((int) keys[keys.length-1]/ (double)instancesByCLA.numInstances())*100.0;
		String additionalSelectedMetricIndices = "";
		
		//int i=0;
		//for(Object key: keys){
			//System.out.println(i++ + ": " + key + "  " + metricIdxWithTheSameViolationScores.get(key) + "  percent: " + (double) ((int) key/ (double)instancesByCLA.numInstances())*100.0);
			
			// if ((double) ((int) key/ (double)instancesByCLA.numInstances())*100.0 >= inversePercent) {
			//if (maxPer >= inversePercent) { 
			//	additionalSelectedMetricIndices += metricIdxWithTheSameViolationScores.get(key);
			//}
		//}
		
		//System.out.println("instance num: " + instancesByCLA.numInstances()+","+keys[keys.length-1]+","+maxPer+","+keys[0]+","+minPer);
		
		Instances trainingInstancesByCLAMI = null;
		
		// (2) Generate instances for CLAMI. If there are no instances in the first round with the minimum violation scores,
		//     then use the next minimum violation score. (Keys are ordered violation scores)
		Instances newTestInstances = null;
		for(Object key: keys){
			if (inversePercent > 0) {
				//String selectedMetricIndices = metricIdxWithTheSameViolationScores.get(key) + (instancesByCLA.classIndex() +1);
				//System.out.println("metric idex1: " + selectedMetricIndices);
				//if (minPer + maxPer >= inversePercent) { 
				String	selectedMetricIndices = metricIdxWithTheSameViolationScores.get(key) + metricIdxWithTheSameViolationScores.get((int)keys[keys.length-1]) + additionalSelectedMetricIndices + (instancesByCLA.classIndex() +1);
				//}
				//System.out.println("metric idex2: " + selectedMetricIndices);
				
				System.out.println("This is [CLAMI+]. Allowed scope is " + inversePercent +"%" );
				trainingInstancesByCLAMI = getInstancesByRemovingSpecificAttributes(instancesByCLA,selectedMetricIndices,true);
				newTestInstances = getInstancesByRemovingSpecificAttributes(testInstances,selectedMetricIndices,true);
				
				System.out.println("number of instances before instance selection: "+ trainingInstancesByCLAMI.numInstances());
						
				// Instance selection
				cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(trainingInstancesByCLAMI,percentileCutoff); // get higher value cutoffs from the metric-selected dataset
				String instIndicesNeedToRemove = getSelectedInstances(trainingInstancesByCLAMI,metricIdxWithTheSameViolationScores.get(key),metricIdxWithTheSameViolationScores.get((int)keys[keys.length-1]), cutoffsForHigherValuesOfAttribute,positiveLabel);
				trainingInstancesByCLAMI = getInstancesByRemovingSpecificInstances(trainingInstancesByCLAMI,instIndicesNeedToRemove,false);
				
				System.out.println("number of instances after instance selection: "+ trainingInstancesByCLAMI.numInstances());
			}
			
			else {
				String selectedMetricIndices = metricIdxWithTheSameViolationScores.get(key) + (instancesByCLA.classIndex() +1); // selectedMetricIndices는 key의 value, 즉, violation에 해당하는 attr들의 index를 갖고 + (class index+1)    ex) " 16, + 10 "
				trainingInstancesByCLAMI = getInstancesByRemovingSpecificAttributes(instancesByCLA,selectedMetricIndices,true); // trainingInstancesByCLAMI는 CLA로 예측한 label을 갖고 있는 것에서, selectedMetricIndices에 있는 attr index에 해당하는 attribute만 남긴 것.  
				newTestInstances = getInstancesByRemovingSpecificAttributes(testInstances,selectedMetricIndices,true);  // 원래 dataset에서, selectedMetricIndices에 있는 attr index에 해당하는 attribute만 남긴 것. 
						
				System.out.println("number of instances before instance selection: "+ trainingInstancesByCLAMI.numInstances());
				
				// Instance selection
				cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(trainingInstancesByCLAMI,percentileCutoff); // get higher value cutoffs from the metric-selected dataset   // metric selection하고 남은 dataset만을 가지고, 그것들의 cutoff에 해당하는 값을 저장. 
				String instIndicesNeedToRemove = getSelectedInstances(trainingInstancesByCLAMI,"","",cutoffsForHigherValuesOfAttribute,positiveLabel);  // violation을 가지고 있는 instance들의 index를 string으로 instIndicesNeedToRemove에 저장한다. (1,2,3이런 형식으로)
				trainingInstancesByCLAMI = getInstancesByRemovingSpecificInstances(trainingInstancesByCLAMI,instIndicesNeedToRemove,false);  // instance 제거하고 남은 dataset을 trainingInstancesByCLAMI에 저장한다. 
				
				System.out.println("number of instances after instance selection: "+ trainingInstancesByCLAMI.numInstances());
				
			}
			if(trainingInstancesByCLAMI.numInstances() != 0)
				break;
		}
		
		// check if there are no instances in any one of two classes.
		if(trainingInstancesByCLAMI.attributeStats(trainingInstancesByCLAMI.classIndex()).nominalCounts[0]!=0 &&
				trainingInstancesByCLAMI.attributeStats(trainingInstancesByCLAMI.classIndex()).nominalCounts[1]!=0){
		
			try {
				Classifier classifier = (Classifier) weka.core.Utils.forName(Classifier.class, mlAlgorithm, null);
				classifier.buildClassifier(trainingInstancesByCLAMI);
				
				// Print CLAMI results
				int TP=0, FP=0,TN=0, FN=0;
				for(int instIdx = 0; instIdx < newTestInstances.numInstances(); instIdx++){
					double[] probability = classifier.distributionForInstance(newTestInstances.get(instIdx));
					
//					for(int i = 0; i < probability.length; i++){
//						System.out.println("Probability of class " + newTestInstances.classAttribute().value(i) + " : " + Double.toString(probability[i]));
//					}
					
					double predictedLabelIdx = (probability[0] >= threshold)? 0.0: 1.0;
					
					if(!suppress)
						System.out.println("CLAMI: Instance " + (instIdx+1) + " predicted as, " + 
							newTestInstances.classAttribute().value((int)predictedLabelIdx)	+
							//((newTestInstances.classAttribute().indexOfValue(positiveLabel))==predictedLabelIdx?"buggy":"clean") +
							", (Actual class: " + Utils.getStringValueOfInstanceLabel(newTestInstances,instIdx) + ") ");
					// compute T/F/P/N for the original instances labeled.
					if(!Double.isNaN(instances.get(instIdx).classValue())){
						if(predictedLabelIdx==instances.get(instIdx).classValue()){
							if(predictedLabelIdx==instances.attribute(instances.classIndex()).indexOfValue(positiveLabel))
								TP++;
							else
								TN++;
						}else{
							if(predictedLabelIdx==instances.attribute(instances.classIndex()).indexOfValue(positiveLabel))
								FP++;
							else
								FN++;
						}
					}
				}
				
				Evaluation eval = new Evaluation(trainingInstancesByCLAMI);
				eval.evaluateModel(classifier, newTestInstances);
				
				if (TP+TN+FP+FN>0){
					printEvaluationResult(TP, TN, FP, FN, experimental);
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

	private static String getSelectedInstances(Instances instances,String notInverseIndex, String inverseIndex, double[] cutoffsForHigherValuesOfAttribute,
			String positiveLabel) {
		//System.out.println("in getSelectedInstances not inverse: "+notInverseIndex + "  inverse: " + inverseIndex);
		int numOfAttrforInverseViolation=0;
		int numOfAttrforNOTInverseViolation=0;
		for (int i=0;i<inverseIndex.length();i++) {
			if (inverseIndex.charAt(i) == ',') {
				numOfAttrforInverseViolation++;
			}
		}
		for (int i=0;i<notInverseIndex.length();i++) {
			if (notInverseIndex.charAt(i) == ',') {
				numOfAttrforNOTInverseViolation++;
			}
		}
		
		int[] violations = new int[instances.numInstances()];
		
		// inverse 되는 violation이 없는 경우 
		if (numOfAttrforInverseViolation == 0) {
			for(int instIdx=0; instIdx < instances.numInstances(); instIdx++){
				
				for(int attrIdx=0; attrIdx < instances.numAttributes(); attrIdx++){
					if(attrIdx == instances.classIndex())
						continue; // no need to compute violation score for the class attribute
					
					if (instances.get(instIdx).value(attrIdx) <= cutoffsForHigherValuesOfAttribute[attrIdx] // cutoff에 해당하는 값보다 작거나 같은데 (clean일 것)
							&& instances.get(instIdx).classValue() == instances.classAttribute().indexOfValue(positiveLabel)){ // class label이 buggy인 경우,
							violations[instIdx]++; // 해당 attribute의 violation 개수++
					}else if(instances.get(instIdx).value(attrIdx) > cutoffsForHigherValuesOfAttribute[attrIdx] // cutoff에 해당하는 값보다 큰데 (buggy일 것)
							&& instances.get(instIdx).classValue() == instances.classAttribute().indexOfValue(getNegLabel(instances, positiveLabel))){ // class label이 clean인 경우, 
							violations[instIdx]++; // 해당 attribute의 violation 개수++ 
					}
				}
			}
		}
		// inverse 되는 violation이 있는 경우 
		else {
			for(int instIdx=0; instIdx < instances.numInstances(); instIdx++){
				
				for(int attrIdx=0; attrIdx < numOfAttrforNOTInverseViolation; attrIdx++){
					if(attrIdx == instances.classIndex())
						continue; // no need to compute violation score for the class attribute
					
					if (instances.get(instIdx).value(attrIdx) <= cutoffsForHigherValuesOfAttribute[attrIdx] // cutoff에 해당하는 값보다 작거나 같은데 (clean일 것)
							&& instances.get(instIdx).classValue() == instances.classAttribute().indexOfValue(positiveLabel)){ // class label이 buggy인 경우,
							violations[instIdx]++; // 해당 attribute의 violation 개수++
					}else if(instances.get(instIdx).value(attrIdx) > cutoffsForHigherValuesOfAttribute[attrIdx] // cutoff에 해당하는 값보다 큰데 (buggy일 것)
							&& instances.get(instIdx).classValue() == instances.classAttribute().indexOfValue(getNegLabel(instances, positiveLabel))){ // class label이 clean인 경우, 
							violations[instIdx]++; // 해당 attribute의 violation 개수++ 
					}
				}
			}
			// violation 반대인 애는 반대로 구하기 
			for(int instIdx=0; instIdx < instances.numInstances(); instIdx++){
				
				for(int attrIdx=numOfAttrforNOTInverseViolation; attrIdx < instances.numAttributes(); attrIdx++){
					if(attrIdx == instances.classIndex())
						continue; // no need to compute violation score for the class attribute
					
					if (instances.get(instIdx).value(attrIdx) > cutoffsForHigherValuesOfAttribute[attrIdx]
							&& instances.get(instIdx).classValue() == instances.classAttribute().indexOfValue(positiveLabel)){
							violations[instIdx]++;
					}else if(instances.get(instIdx).value(attrIdx) <= cutoffsForHigherValuesOfAttribute[attrIdx]
							&& instances.get(instIdx).classValue() == instances.classAttribute().indexOfValue(getNegLabel(instances, positiveLabel))){
							violations[instIdx]++;
					}
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
	 * @throws Exception 
	 */
	public static Instances loadArff(String path,String classAttributeName) throws Exception{
		fileName = path;
		Instances instances=null;
		BufferedReader reader;
		try {
			reader = new BufferedReader(new FileReader(path));
//			instances = new Instances(reader);
			DataSource source = new DataSource(path);
			instances = source.getDataSet();
//			reader.close();
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