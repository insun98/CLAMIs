package net.lifove.clami;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import net.lifove.clami.util.Utils;
import weka.classifiers.Classifier;
import weka.core.Instances;

public class CLABI implements ICLAMI {

	protected Instances trainingInstances;
	protected Instances testInstances;
	protected static Instances instancesByCLA;
	protected String mlAlg;
	boolean isExperimental;
	protected HashMap<Integer, String> metricIdxWithTheSameViolationScores;
	protected static List<Double> probabilityOfCLAMIIdx;
	protected static List<Double> probabilityOfCLABIIdx;
	protected static List<Double> CLAMIIdx;
	protected static List<Double> CLABIIdx;
	List<Double> probabilityOfIdx = new ArrayList<Double>();
	List<Double> predictedIdx = new ArrayList<Double>();
	Classifier classifier;
	
	/**
	 * Constructor
	 * @param mlAlg: machine learning algorithm
	 * @param isExperimental: to check if experiment option is on
	 */
	CLABI(String mlAlg, boolean isExperimental) {
		trainingInstances = null;
		testInstances = null;
		instancesByCLA = null;
		this.mlAlg = mlAlg;
		this.isExperimental = isExperimental;
		metricIdxWithTheSameViolationScores = null;
		probabilityOfCLAMIIdx = new ArrayList<Double>();
		probabilityOfCLABIIdx = new ArrayList<Double>();
		CLAMIIdx = new ArrayList<Double>();
		CLABIIdx = new ArrayList<Double>();
	}
	
	/**
	 * Get CLABI result
	 * @param instances
	 * @param percentileCutoff cutoff percentile for top and bottom clusters
	 * @param positiveLabel positive label string value
	 * @param suppress detailed prediction results
	 * @param isDegree: to get if clustering has to done with continuous values
	 * @param fileName: name of the running file
	 * @return instances labeled by CLAMI
	 */
	public void getResult(Instances instances, double percentileCutoff, String positiveLabel, boolean suppress,
			boolean isDegree, String fileName) {
		getResult(instances, percentileCutoff, positiveLabel, suppress, false, isDegree, fileName);
	}
	
	/**
	 * Get CLABI result
	 * @param instances
	 * @param percentileCutoff cutoff percentile for top and bottom clusters
	 * @param positiveLabel positive label string value
	 * @param suppress detailed prediction results
	 * @param isDegree: to get if clustering has to done with continuous values
	 * @param fileName: name of the running file
	 * @return instances labeled by CLAMI
	 */
	public void getResult(Instances instances, double percentileCutoff, String positiveLabel, boolean suppress,
			boolean experimental, boolean isDegree, String fileName) {
		clustering(instances, percentileCutoff, positiveLabel);

		double[] cutoffsForHigherValuesOfAttribute = Utils.getHigherValueCutoffs(instancesByCLA, percentileCutoff);

		metricIdxWithTheSameViolationScores = Utils.getMetricIndicesWithTheViolationScores(instancesByCLA,
				cutoffsForHigherValuesOfAttribute, positiveLabel);

		Object[] keys = metricIdxWithTheSameViolationScores.keySet().toArray();
		Object[] descending_keys = metricIdxWithTheSameViolationScores.keySet().toArray();

		Arrays.sort(descending_keys, Collections.reverseOrder());
		getTrainingTestSet(descending_keys, instances, positiveLabel, percentileCutoff);
		getProbabiltyOfIdx();
		probabilityOfCLABIIdx.addAll(probabilityOfIdx);

		CLABIIdx.addAll(predictedIdx);

		if (CLABIIdx == null || probabilityOfCLABIIdx == null) {
			CLAMI clami = new CLAMI(mlAlg, isExperimental);
			clami.getResult(instances, percentileCutoff, positiveLabel, suppress, isDegree, fileName);
			return;

		}

		Arrays.sort(keys);
		getTrainingTestSet(keys, instances, positiveLabel, percentileCutoff);
		getProbabiltyOfIdx();
		probabilityOfCLAMIIdx.addAll(probabilityOfIdx);
		CLAMIIdx.addAll(predictedIdx);

		getLabeling(instances, positiveLabel);
		getPredictedLabels(suppress, instances);
		printResult(instances, experimental, fileName, suppress, positiveLabel);
	}
	
	/**
	 * To do clustering
	 * @param instances
	 * @param percentileCutoff cutoff percentile for top and bottom clusters
	 * @param positiveLabel positive label string value
	 */
	public Instances clustering(Instances instances, double percentileCutoff, String positiveLabel) {
		CLA cla = new CLA();
		instancesByCLA = cla.clustering(instances, percentileCutoff, positiveLabel);
		return null;
	}
	
	/**
	 * To do clustering
	 * @param instances
	 * @param percentileCutoff cutoff percentile for top and bottom clusters
	 * @param positiveLabel positive label string value
	 */

	public void getTrainingTestSet(Object[] keys, Instances instances, String positiveLabel, double percentileCutoff) {
		trainingInstances = null;
		testInstances = null;

		for (Object key : keys) {

			String selectedMetricIndices = metricIdxWithTheSameViolationScores.get(key)
					+ (instancesByCLA.classIndex() + 1);
			trainingInstances = Utils.getInstancesByRemovingSpecificAttributes(instancesByCLA, selectedMetricIndices,
					true);
			testInstances = Utils.getInstancesByRemovingSpecificAttributes(instances, selectedMetricIndices, true);

			// Instance selection
			double[] cutoffsForHigherValuesOfAttribute = Utils.getHigherValueCutoffs(trainingInstances,
					percentileCutoff); // get higher value cutoffs from the metric-selected dataset
			String instIndicesNeedToRemove = Utils.getSelectedInstances(trainingInstances,
					cutoffsForHigherValuesOfAttribute, positiveLabel);
			trainingInstances = Utils.getInstancesByRemovingSpecificInstances(trainingInstances,
					instIndicesNeedToRemove, false);

			if (trainingInstances.numInstances() != 0)
				break;
		}

		if (trainingInstances.attributeStats(trainingInstances.classIndex()).nominalCounts[0] != 0
				&& trainingInstances.attributeStats(trainingInstances.classIndex()).nominalCounts[1] != 0)
			return;
		else
			System.err.println(
					"Dataset is not proper to build a CLAMI model! Dataset does not follow the assumption, i.e. the higher metric value, the more bug-prone.");
	}
	
	/**
	 * Get predicted index and the probability of it
	 */
	public void getProbabiltyOfIdx() {
		probabilityOfIdx.removeAll(probabilityOfIdx);
		predictedIdx.removeAll(predictedIdx);
		double[] prediction;
		String mlAlgorithm = mlAlg != null && !mlAlg.equals("") ? mlAlg : "weka.classifiers.functions.Logistic";

		try {
			classifier = (Classifier) weka.core.Utils.forName(Classifier.class, mlAlgorithm, null);
			classifier.buildClassifier(trainingInstances);

			for (int instIdx = 0; instIdx < testInstances.numInstances(); instIdx++) {
				double LabelIdx = classifier.classifyInstance(testInstances.get(instIdx));
				predictedIdx.add(LabelIdx);

				prediction = classifier.distributionForInstance(testInstances.get(instIdx)); // probability of clean and
																								// buggy

				double max = prediction[0]; // take first index of prediction as max

				for (int i = 0; i < prediction.length; i++) {

					if (max < prediction[i])
						max = prediction[i]; // find max
				}

				probabilityOfIdx.add(max);
			}
		} catch (Exception e) {
			System.err.println(
					"Specify the correct Weka machine learing classifier with a fully qualified name. E.g., weka.classifiers.functions.Logistic");
			e.printStackTrace();
			System.exit(0);
		}
	}
	
	/**
	 * Get Final TrainingModel
	 * @param instances
	 * @param positiveLabel
	 */
	public static void getLabeling(Instances instances, String positiveLabel) {

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

	}

	/**
	 * Get Final predicted labels
	 * @param suppress
	 * @param get final labeling
	 */
	public void getPredictedLabels(boolean suppress, Instances instances) {

		String mlAlgorithm = mlAlg != null && !mlAlg.equals("") ? mlAlg : "weka.classifiers.functions.Logistic";

		try {
			classifier = (Classifier) weka.core.Utils.forName(Classifier.class, mlAlgorithm, null);
			classifier.buildClassifier(instancesByCLA);

			for (int instIdx = 0; instIdx < instances.numInstances(); instIdx++) {
				double LabelIdx = classifier.classifyInstance(instances.get(instIdx));
				if (!suppress)
					System.out.println("CLAMI: Instance " + (instIdx + 1) + " predicted as, "
							+ instances.classAttribute().value((int) LabelIdx) +
							", (Actual class: " + Utils.getStringValueOfInstanceLabel(instances, instIdx) + ") ");

			}

		} catch (Exception e) {
			System.err.println(
					"Specify the correct Weka machine learing classifier with a fully qualified name. E.g., weka.classifiers.functions.Logistic");
			e.printStackTrace();
			System.exit(0);
		}
	}
	
	/**
	 * Get the result printed
	 * @param instances
	 * @param isExperimental: to check if experiment option is on
	 * @param fileName: name of the running file
	 * @param suppress detailed prediction results
	 * @param positiveLabel positive label string value
	 */
	public void printResult(Instances instances, boolean experimental, String fileName, boolean suppress,
			String positiveLabel) {
		Utils.printEvaluationResult(instances, instances, instancesByCLA, classifier, positiveLabel, experimental,
				fileName);

	}
}