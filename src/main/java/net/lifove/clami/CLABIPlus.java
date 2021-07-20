package net.lifove.clami;

import java.util.Arrays;
import java.util.Collections;
import net.lifove.clami.util.Utils;
import weka.core.Instances;

public class CLABIPlus extends CLABI implements ICLAMI {

	/**
	 * Constructor
	 * @param mlAlg: machine learning algorithm
	 * @param isExperimental: to check if experiment option is on
	 */
	CLABIPlus(String mlAlg, boolean isExperimental) {
		super(mlAlg, isExperimental);
		// TODO Auto-generated constructor stub
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
		CLAPlus claPlus = new CLAPlus();
		instancesByCLA = claPlus.clustering(instances, percentileCutoff, positiveLabel);
		return null;
	}
	
}
