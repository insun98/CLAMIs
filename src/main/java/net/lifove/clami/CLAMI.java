package net.lifove.clami;

import java.util.Arrays;
import java.util.HashMap;
import net.lifove.clami.util.Utils;
import weka.classifiers.Classifier;
import weka.core.Instances;

public class CLAMI implements ICLAMI {

	private Instances trainingInstances;
	private Instances testInstances;
	private CLA cla = new CLA();
	private Instances instancesByCLA;
	private String mlAlg;
	boolean isExperimental;
	private HashMap<Integer, String> metricIdxWithTheSameViolationScores;
	Classifier classifier;

	CLAMI(String mlAlg, boolean isExperimental) {
		trainingInstances = null;
		testInstances = null;
		instancesByCLA = null;
		this.mlAlg = mlAlg;
		this.isExperimental = isExperimental;
		metricIdxWithTheSameViolationScores = null;
	}

	public void getResult(Instances instances, double percentileCutoff, String positiveLabel, boolean suppress,
			boolean isDegree, String fileName) {
		getResult(instances, percentileCutoff, positiveLabel, suppress, false, isDegree, fileName);
	}

	public void getResult(Instances instances, double percentileCutoff, String positiveLabel, boolean suppress,
			boolean experimental, boolean isDegree, String fileName) {
		instancesByCLA = new Instances(instances);
		if (isDegree)
			clusteringForContinuousValue(instances, percentileCutoff, positiveLabel);
		else
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

	public Instances clustering(Instances instances, double percentileCutoff, String positiveLabel) {
		instancesByCLA = cla.clustering(instances, percentileCutoff, positiveLabel);
		return null;
	}

	public Instances clusteringForContinuousValue(Instances instances, double percentileCutoff, String positiveLabel) {
		instancesByCLA = cla.clusteringForContinuousValue(instances, percentileCutoff, positiveLabel);
		return null;
	}

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

	public void printResult(Instances instances, boolean experimental, String fileName, boolean suppress,
			String positiveLabel) {
		Utils.printEvaluationResult(instances, testInstances, trainingInstances, classifier, positiveLabel,
				experimental, fileName);

	}

}
