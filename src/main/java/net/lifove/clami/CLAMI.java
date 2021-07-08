package net.lifove.clami;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import net.lifove.clami.util.Utils;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

public class CLAMI {
	
	static List<Double> probabilityOfIdx = new ArrayList<Double>(); 
	static List<Double> predictedLabelIdx = new ArrayList<Double>();
	
	/**
	 * Get CLAMI result. Since CLAMI is the later steps of CLA, to get instancesByCLA use getCLAResult.
	 * @param testInstances
	 * @param instancesByCLA
	 * @param positiveLabel
	 */
	public static void getCLAMIResult(Instances testInstances, Instances instances, String positiveLabel,double percentileCutoff,boolean suppress,String mlAlg, boolean isDegree, int sort, boolean forCLABI) {
		getCLAMIResult(testInstances,instances,positiveLabel,percentileCutoff,suppress,false,mlAlg, isDegree, sort, forCLABI); //no experimental as default
	}
	
	public static void getCLAMIResult(Instances testInstances, Instances instances, String positiveLabel,double percentileCutoff, boolean suppress, boolean experimental, String mlAlg, boolean isDegree, int sort, boolean forCLABI) {
		

		probabilityOfIdx.removeAll(probabilityOfIdx) ;
		predictedLabelIdx.removeAll(predictedLabelIdx) ;
		
		String mlAlgorithm = mlAlg!=null && !mlAlg.equals("")?mlAlg:"weka.classifiers.functions.Logistic";
		
		Instances instancesByCLA = Utils.getInstancesByCLA(instances, percentileCutoff, positiveLabel, isDegree);
		
		// Compute medians
		double[] cutoffsForHigherValuesOfAttribute = Utils.getHigherValueCutoffs(instancesByCLA,percentileCutoff);
				
		// Metric selection
		
		// (1) get distinct violation scores ordered by ASC
		HashMap<Integer,String> metricIdxWithTheSameViolationScores = Utils.getMetricIndicesWithTheViolationScores(instancesByCLA,cutoffsForHigherValuesOfAttribute,positiveLabel);
		Object[] keys =  metricIdxWithTheSameViolationScores.keySet().toArray();
		if(sort==0) 
			Arrays.sort(keys);
		else 
			Arrays.sort(keys, Collections.reverseOrder());
		
		
		Instances trainingInstancesByCLAMI = null;
		
		// (2) Generate instances for CLAMI. If there are no instances in the first round with the minimum violation scores,
		//     then use the next minimum violation score. (Keys are ordered violation scores)
		Instances newTestInstances = null;
		for(Object key: keys){
			
			String selectedMetricIndices = metricIdxWithTheSameViolationScores.get(key) + (instancesByCLA.classIndex() +1);
			trainingInstancesByCLAMI = Utils.getInstancesByRemovingSpecificAttributes(instancesByCLA,selectedMetricIndices,true);
			newTestInstances = Utils.getInstancesByRemovingSpecificAttributes(testInstances,selectedMetricIndices,true);
					
			// Instance selection
			cutoffsForHigherValuesOfAttribute = Utils.getHigherValueCutoffs(trainingInstancesByCLAMI,percentileCutoff); // get higher value cutoffs from the metric-selected dataset
			String instIndicesNeedToRemove = Utils.getSelectedInstances(trainingInstancesByCLAMI,cutoffsForHigherValuesOfAttribute,positiveLabel);
			trainingInstancesByCLAMI = Utils.getInstancesByRemovingSpecificInstances(trainingInstancesByCLAMI,instIndicesNeedToRemove,false);
			
			if(trainingInstancesByCLAMI.numInstances() != 0)
				break;
		}
		
		
		double[] prediction;
		if(trainingInstancesByCLAMI != null) {
		// check if there are no instances in any one of two classes.
		if(trainingInstancesByCLAMI.attributeStats(trainingInstancesByCLAMI.classIndex()).nominalCounts[0]!=0 &&
				trainingInstancesByCLAMI.attributeStats(trainingInstancesByCLAMI.classIndex()).nominalCounts[1]!=0){
		
			try {
				Classifier classifier = (Classifier) weka.core.Utils.forName(Classifier.class, mlAlgorithm, null);
				classifier.buildClassifier(trainingInstancesByCLAMI);
				
				// Print CLAMI results
				int TP=0, FP=0,TN=0, FN=0;
				for(int instIdx = 0; instIdx < newTestInstances.numInstances(); instIdx++){
					double LabelIdx = classifier.classifyInstance(newTestInstances.get(instIdx));
					predictedLabelIdx.add(LabelIdx);
					
					
					if(!suppress && !forCLABI)
						System.out.println("CLAMI: Instance " + (instIdx+1) + " predicted as, " + 
							newTestInstances.classAttribute().value((int)LabelIdx)	+
							//((newTestInstances.classAttribute().indexOfValue(positiveLabel))==predictedLabelIdx?"buggy":"clean") +
							", (Actual class: " + Utils.getStringValueOfInstanceLabel(newTestInstances,instIdx) + ") ");
					// compute T/F/P/N for the original instances labeled.
					prediction = classifier.distributionForInstance(newTestInstances.get(instIdx)); //probability of clean and buggy
					
					double max = prediction[0]; // take first index of prediction as max  

					for(int i = 0; i < prediction.length; i++){

						if(max < prediction[i]) max = prediction[i]; // find max
					}
					
					probabilityOfIdx.add(max);
					
					if(!Double.isNaN(instances.get(instIdx).classValue())){
						if(LabelIdx==instances.get(instIdx).classValue()){
							if(LabelIdx==instances.attribute(instances.classIndex()).indexOfValue(positiveLabel))
								TP++;
							else
								TN++;
						}else{
							if(LabelIdx==instances.attribute(instances.classIndex()).indexOfValue(positiveLabel))
								FP++;
							else
								FN++;
						}
					}
				}
				
				Evaluation eval = new Evaluation(trainingInstancesByCLAMI);
				eval.evaluateModel(classifier, newTestInstances);
				if(!forCLABI) {
				if (TP+TN+FP+FN>0){
					Utils.printEvaluationResult(TP, TN, FP, FN, eval, newTestInstances, positiveLabel, experimental);
					
				}
				else if(suppress)
					System.out.println("No labeled instances in the arff file. To see detailed prediction results, try again without the suppress option  (-s,--suppress)");
				}
			} catch (Exception e) {
				System.err.println("Specify the correct Weka machine learing classifier with a fully qualified name. E.g., weka.classifiers.functions.Logistic");
				e.printStackTrace();
				System.exit(0);
			}
		}else{
			System.err.println("Dataset is not proper to build a CLAMI model! Dataset does not follow the assumption, i.e. the higher metric value, the more bug-prone.");
		}
			
		}else {
			probabilityOfIdx = null;
			predictedLabelIdx=null;
		}
		
		
	
	}
	
}


