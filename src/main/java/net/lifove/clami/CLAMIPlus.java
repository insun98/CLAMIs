package net.lifove.clami;

import java.util.Arrays;
import net.lifove.clami.util.Utils;
import weka.core.Instances;

/**
 * This class run for CLAMI+. 
 */
public class CLAMIPlus extends CLAMI implements ICLAMI {
	CLAMIPlus(String mlAlg, boolean isExperimental) {
		super(mlAlg, isExperimental);
		// TODO Auto-generated constructor stub
	}

	/**
	 * Get CLAMI result
	 * @param instances
	 * @param percentileCutoff cutoff percentile for top and bottom clusters
	 * @param positiveLabel positive label string value
	 * @param suppress detailed prediction results
	 * @param filePath: name of the running file
	 * @return instances labeled by CLAMI
	 */
	public void getResult(Instances instances, double percentileCutoff, String positiveLabel, boolean suppress,
			boolean isDegree, String filePath) {
		getResult(instances, percentileCutoff, positiveLabel, suppress, false, isDegree, filePath);
	}
	
	/**
	 * Get CLAMI result
	 * @param instances
	 * @param percentileCutoff cutoff percentile for top and bottom clusters
	 * @param positiveLabel positive label string value
	 * @param suppress detailed prediction results
	 * @param filePath: name of the running file
	 * @return instances labeled by CLAMI
	 */
	public void getResult(Instances instances, double percentileCutoff, String positiveLabel, boolean suppress,
			boolean experimental, boolean isDegree, String filePath) {
		instancesByCLA = new Instances(instances);
		
		clustering(instances, percentileCutoff, positiveLabel);

		double[] cutoffsForHigherValuesOfAttribute = Utils.getHigherValueCutoffs(instancesByCLA, percentileCutoff);

		metricIdxWithTheSameViolationScores = Utils.getMetricIndicesWithTheViolationScores(instancesByCLA,
				cutoffsForHigherValuesOfAttribute, positiveLabel);
		Object[] keys = metricIdxWithTheSameViolationScores.keySet().toArray();

		Arrays.sort(keys);

		getCLAMITrainingSet(keys, instances, positiveLabel, percentileCutoff);
		getPredictedLabels(suppress, instances);
		printResult(instances, experimental, filePath, suppress, positiveLabel);
	}
	
	/**
	 * To do clustering
	 * @param instances
	 * @param percentileCutoff cutoff percentile for top and bottom clusters
	 * @param positiveLabel positive label string value
	 */
	@Override
	public Instances clustering(Instances instances, double percentileCutoff, String positiveLabel) {
		CLAPlus claPlus = new CLAPlus();
		instancesByCLA = claPlus.clustering(instances, percentileCutoff, positiveLabel);
		return null;
	}
	
}