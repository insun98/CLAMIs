package net.lifove.clami;

import java.util.Arrays;
import java.util.HashMap;
import net.lifove.clami.util.Utils;
import weka.classifiers.Classifier;
import weka.core.Instances;

public class CLAMI implements ICLAMI {
	protected Instances trainingInstances;
	protected Instances testInstances;
	protected Instances instancesByCLA;
	protected String mlAlg;
	boolean isExperimental;
	protected HashMap<Integer, String> metricIdxWithTheSameViolationScores;
	Classifier classifier;
	
	/**
	 * Constructor
	 * @param mlAlg: machine learning algorithm
	 * @param isExperimental: to check if experiment option is on
	 */
	CLAMI(String mlAlg, boolean isExperimental) {
		trainingInstances = null;
		testInstances = null;
		instancesByCLA = null;
		this.mlAlg = mlAlg;
		this.isExperimental = isExperimental;
		metricIdxWithTheSameViolationScores = null;
	}
	
	/**
	 * Get CLAMI result
	 * @param instances
	 * @param percentileCutoff cutoff percentile for top and bottom clusters
	 * @param positiveLabel positive label string value
	 * @param suppress detailed prediction results
	 * @param fileName: name of the running file
	 * @return instances labeled by CLAMI
	 */
	public void getResult(Instances instances, double percentileCutoff, String positiveLabel, boolean suppress, String fileName) {
		getResult(instances, percentileCutoff, positiveLabel, suppress, false, fileName);
	}
	
	/**
	 * Get CLAMI result
	 * @param instances
	 * @param percentileCutoff cutoff percentile for top and bottom clusters
	 * @param positiveLabel positive label string value
	 * @param suppress detailed prediction results
	 * @param fileName: name of the running file
	 * @return instances labeled by CLAMI
	 */
	public void getResult(Instances instances, double percentileCutoff, String positiveLabel, boolean suppress,
			boolean experimental, String fileName) {
		instancesByCLA = new Instances(instances);
		
		clustering(instances, percentileCutoff, positiveLabel);

		double[] cutoffsForHigherValuesOfAttribute = Utils.getHigherValueCutoffs(instancesByCLA, percentileCutoff);

		metricIdxWithTheSameViolationScores = Utils.getMetricIndicesWithTheViolationScores(instancesByCLA,
				cutoffsForHigherValuesOfAttribute, positiveLabel);
		Object[] keys = metricIdxWithTheSameViolationScores.keySet().toArray();

		Arrays.sort(keys);

		getTrainingTestSet(keys, instances, positiveLabel, percentileCutoff);
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
	 * Get Training and Test Set after metric and instance selection 
	 * @param keys: MVS
	 * @param instances
	 * @param percentileCutoff cutoff percentile for top and bottom clusters
	 * @param positiveLabel positive label string value
	 */
	public void getTrainingTestSet(Object[] keys, Instances instances, String positiveLabel, double percentileCutoff) {

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
	 * Get Labeling
	 * @param instances
	 * @param suppress
	 */
	public void getPredictedLabels(boolean suppress, Instances instances) {
		String mlAlgorithm = mlAlg != null && !mlAlg.equals("") ? mlAlg : "weka.classifiers.functions.Logistic";

		try {
			classifier = (Classifier) weka.core.Utils.forName(Classifier.class, mlAlgorithm, null);
			classifier.buildClassifier(trainingInstances);

			for (int instIdx = 0; instIdx < testInstances.numInstances(); instIdx++) {
				double LabelIdx = classifier.classifyInstance(testInstances.get(instIdx));
				if (!suppress)
					System.out.println("CLAMI: Instance " + (instIdx + 1) + " predicted as, "
							+ testInstances.classAttribute().value((int) LabelIdx) +
							", (Actual class: " + Utils.getStringValueOfInstanceLabel(testInstances, instIdx) + ") ");

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
		Utils.printEvaluationResult(instances, testInstances, trainingInstances, classifier, positiveLabel,
				experimental, fileName);

	}

}
