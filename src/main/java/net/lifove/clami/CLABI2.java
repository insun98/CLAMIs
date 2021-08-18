 package net.lifove.clami;


import java.util.Arrays;
import java.util.Collections;
import net.lifove.clami.util.Utils;
import weka.core.Instances;

public class CLABI2 extends CLAMI implements ICLAMI {

	
	CLABI2(String mlAlg, boolean isExperimental) {
		super(mlAlg, isExperimental);
	}
	
	/**
	 * Get CLABI result
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
	 * To do clustering
	 * @param instances
	 * @param percentileCutoff cutoff percentile for top and bottom clusters
	 * @param positiveLabel positive label string value
	 */
	public void getCLAMITrainingSet(Object[] keys, Instances instances, String positiveLabel, double percentileCutoff) {
		Object[] descending_keys = metricIdxWithTheSameViolationScores.keySet().toArray();
		Arrays.sort(descending_keys, Collections.reverseOrder());
		String selectedMetricIndices = null;
		String instIndicesNeedToRemove = null;
		int i = 0;
		for (Object key : keys) {

			selectedMetricIndices = metricIdxWithTheSameViolationScores.get(key)+metricIdxWithTheSameViolationScores.get(descending_keys[i++])
					+ (instancesByCLA.classIndex() + 1);

			String [] splited = metricIdxWithTheSameViolationScores.get(key).split(",");
			
			int  inversedMetricIndex= splited.length;

			trainingInstances = Utils.getInstancesByRemovingSpecificAttributes(instancesByCLA, selectedMetricIndices,
					true);
			testInstances = Utils.getInstancesByRemovingSpecificAttributes(instances, selectedMetricIndices, true);

			// Instance selection
			double[] cutoffsForHigherValuesOfAttribute = Utils.getHigherValueCutoffs(trainingInstances,
					percentileCutoff); // get higher value cutoffs from the metric-selected dataset
			instIndicesNeedToRemove = Utils.getSelectedInstances(trainingInstances,
					cutoffsForHigherValuesOfAttribute, positiveLabel, inversedMetricIndex);
			trainingInstances = Utils.getInstancesByRemovingSpecificInstances(trainingInstances,
					instIndicesNeedToRemove, false);

			if (trainingInstances.numInstances() != 0)
				break;
		}

		if (trainingInstances.attributeStats(trainingInstances.classIndex()).nominalCounts[0] != 0
				&& trainingInstances.attributeStats(trainingInstances.classIndex()).nominalCounts[1] != 0){
			System.out.println("Removed Instances CLABI: "+ instIndicesNeedToRemove);
			System.out.println("Selected Metrics CLABI: "+ selectedMetricIndices);
			return;
		}
		else
			System.err.println(
					"Dataset is not proper to build a CLAMI model! Dataset does not follow the assumption, i.e. the higher metric value, the more bug-prone.");
	}
	
}