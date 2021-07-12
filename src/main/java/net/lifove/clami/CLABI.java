package net.lifove.clami;

import java.util.ArrayList;
import java.util.List;

import net.lifove.clami.util.Utils;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

public class CLABI {
	
	private static List<Double> probabilityOfCLAMIIdx = new ArrayList<Double>();
	private static List<Double> probabilityOfCLABIIdx = new ArrayList<Double>();
	private static List<Double> CLAMIIdx = new ArrayList<Double>();
	private static List<Double> CLABIIdx = new ArrayList<Double>();

	/**
	 * Get CLAMI result. Since CLAMI is the later steps of CLA, to get
	 * instancesByCLA use getCLAResult.
	 * 
	 * @param testInstances
	 * @param instancesByCLA
	 * @param positiveLabel
	 */
	public static void getCLABIResult(Instances testInstances, Instances instances, String positiveLabel,
			double percentileCutoff, boolean suppress, String mlAlg, boolean isDegree, int sort, boolean forCLABI,
			String fileName) {
		getCLABIResult(testInstances, instances, positiveLabel, percentileCutoff, suppress, false, mlAlg, isDegree,
				sort, forCLABI, fileName); // no experimental as default

	}

	/**
	 * Get CLAMI result. Since CLAMI is the later steps of CLA, to get
	 * instancesByCLA use getCLAResult.
	 * 
	 * @param testInstances
	 * @param instancesByCLA
	 * @param positiveLabel
	 */
	public static void getCLABIResult(Instances testInstances, Instances instances, String positiveLabel,
			double percentileCutoff, boolean suppress, boolean experimental, String mlAlg, boolean isDegree, int sort,
			boolean forCLABI, String fileName) {
		probabilityOfCLAMIIdx.removeAll(probabilityOfCLAMIIdx) ;
		probabilityOfCLABIIdx.removeAll(probabilityOfCLABIIdx) ;
		CLAMIIdx.removeAll(CLAMIIdx) ;
		CLABIIdx.removeAll(CLABIIdx) ;
		Instances instancesByCLA = Utils.getInstancesByCLA(instances, percentileCutoff, positiveLabel, isDegree);
		Instances final_newTestInstances = instancesByCLA;

		// call getCLAMIResult with Descending order (sort = 1)
		CLAMI.getCLAMIResult(testInstances, instances, positiveLabel, percentileCutoff, suppress, experimental, mlAlg,
				isDegree, 1, forCLABI, fileName);
		CLABIIdx.addAll(CLAMI.predictedLabelIdx);
		probabilityOfCLABIIdx.addAll(CLAMI.probabilityOfIdx);
		
//		System.out.println("=========CLABIIdx==========");
//		for (int i=0;i<CLABIIdx.size();i++) {
//			System.out.println(CLABIIdx.get(i));
//		}
//		
//		System.out.println("=========probabilityOfCLABIIdx==========");
//		for (int i=0;i<probabilityOfCLABIIdx.size();i++) {
//			System.out.println(probabilityOfCLABIIdx.get(i));
//		}
//		

		// if Descending result is null, just execute CLAMI and return
		if (CLABIIdx == null || probabilityOfCLABIIdx == null) {
			CLAMI.getCLAMIResult(testInstances, instances, positiveLabel, percentileCutoff, suppress, experimental,
					mlAlg, isDegree, 0, false, fileName);
			return;

		}
		// else if Descending result is not null, Execute CLABI
		else {
			// call getCLAMIResult with Ascending order (sort = 0)
			CLAMI.getCLAMIResult(testInstances, instances, positiveLabel, percentileCutoff, suppress, experimental,
					mlAlg, isDegree, 0, forCLABI, fileName);
			CLAMIIdx.addAll(CLAMI.predictedLabelIdx);
			probabilityOfCLAMIIdx.addAll(CLAMI.probabilityOfIdx);
			
//			System.out.println("=========CLAMIIdx==========");
//			for (int i=0;i<CLAMIIdx.size();i++) {
//				System.out.println(CLAMIIdx.get(i));
//			}
//			
//			System.out.println("=========probabilityOfCLAMIIdx==========");
//			for (int i=0;i<probabilityOfCLAMIIdx.size();i++) {
//				System.out.println(probabilityOfCLAMIIdx.get(i));
//			}
			
			

			Instances labeling = getLabeling(final_newTestInstances, CLAMIIdx, CLABIIdx, probabilityOfCLAMIIdx,
					probabilityOfCLABIIdx, positiveLabel);
			
//			System.out.println("==========labeling==========");
//			for (int i=0;i<labeling.size();i++) {
//				System.out.println(labeling.get(i).classValue());
//			}
			
			

			int TP = 0, FP = 0, TN = 0, FN = 0;

			// double[] final_prediction;
			String mlAlgorithm = mlAlg != null && !mlAlg.equals("") ? mlAlg : "weka.classifiers.functions.Logistic";
			// check if there are no instances in any one of two classes.
			if(labeling.attributeStats(labeling.classIndex()).nominalCounts[0]!=0 &&
					labeling.attributeStats(labeling.classIndex()).nominalCounts[1]!=0){
			
				try {
					Classifier final_classifier = (Classifier) weka.core.Utils.forName(Classifier.class, mlAlgorithm, null);
					final_classifier.buildClassifier(labeling);
					
					// Print CLAMI results
					
					for(int instIdx = 0; instIdx < final_newTestInstances.numInstances(); instIdx++){
						double final_predictedLabelIdx = final_classifier.classifyInstance(final_newTestInstances.get(instIdx));
					System.out.println("final predicted Label Index : " + final_predictedLabelIdx);
						
						
						if(!suppress) {
							System.out.println("CLAMI: Instance " + (instIdx+1) + " predicted as, " + 
									final_newTestInstances.classAttribute().value((int)final_predictedLabelIdx)	+
									//((newTestInstances.classAttribute().indexOfValue(positiveLabel))==predictedLabelIdx?"buggy":"clean") +
									", (Actual class: " + Utils.getStringValueOfInstanceLabel(final_newTestInstances,instIdx) + ") ");
						}
						
						
					
						// compute T/F/P/N for the original instances labeled.
						if(!Double.isNaN(instances.get(instIdx).classValue())){
							
							if(final_predictedLabelIdx==instances.get(instIdx).classValue()){
								if(final_predictedLabelIdx==instances.attribute(instances.classIndex()).indexOfValue(positiveLabel)) {
									TP++;
								}
								else {
									TN++;
								}
							}else{
								if(final_predictedLabelIdx==instances.attribute(instances.classIndex()).indexOfValue(positiveLabel)) {
									FP++;
								}
								else {
									FN++;
								}
							}
						}
					}
					
					Evaluation eval = new Evaluation(labeling);
					eval.evaluateModel(final_classifier, final_newTestInstances);

					if (TP + TN + FP + FN > 0) {
						Utils.printEvaluationResult(TP, TN, FP, FN, eval, final_newTestInstances, positiveLabel, experimental,
								fileName);
					}

					else if (suppress)
						System.out.println(
								"No labeled instances in the arff file. To see detailed prediction results, try again without the suppress option  (-s,--suppress)");

				} catch (Exception e) {
					System.err.println(
							"Specify the correct Weka machine learing classifier with a fully qualified name. E.g., weka.classifiers.functions.Logistic");
					e.printStackTrace();
					System.exit(0);
				}
			} else {
				System.err.println(
						"Dataset is not proper to build a CLAMI model! Dataset does not follow the assumption, i.e. the higher metric value, the more bug-prone.");
			}

		}

	}

	/***
	 * To solve the conflict between ascending orders label result and descending
	 * order label result
	 * 
	 * @param instances
	 * @param CLAMIIdx
	 * @param CLABIIdx
	 * @param probabilityOfCLAMIIdx
	 * @param probabilityOfCLABIIdx
	 * @param positiveLabel
	 * @return
	 */
	private static Instances getLabeling(Instances instances, List<Double> CLAMIIdx, List<Double> CLABIIdx,
			List<Double> probabilityOfCLAMIIdx, List<Double> probabilityOfCLABIIdx, String positiveLabel) {

		Instances instancesByCLA = new Instances(instances);

		for (int instIdx = 0; instIdx < instances.numInstances(); instIdx++) {

			String negativeLabel = Utils.getNegLabel(instancesByCLA, positiveLabel);

			if (!(CLAMIIdx.get(instIdx).equals(CLABIIdx.get(instIdx)))) {

				if (probabilityOfCLAMIIdx.get(instIdx) < probabilityOfCLABIIdx.get(instIdx)) {
					if (instances.attribute(instances.classIndex())
							.indexOfValue(positiveLabel) == (CLAMIIdx.get(instIdx))) {
						instancesByCLA.instance(instIdx)
								.setClassValue(Utils.getNegLabel(instancesByCLA, positiveLabel));

					} else if (instances.attribute(instances.classIndex())
							.indexOfValue(negativeLabel) == (CLAMIIdx.get(instIdx))) {
						instancesByCLA.instance(instIdx).setClassValue(positiveLabel);
					}

				} else {
					instancesByCLA.instance(instIdx).setClassValue(CLAMIIdx.get(instIdx));
				}
			} else {
				instancesByCLA.instance(instIdx).setClassValue(CLAMIIdx.get(instIdx));
			}
		}

		return instancesByCLA;
	}
}