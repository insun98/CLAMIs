package net.lifove.clami;

import weka.core.Instances;

public interface ICLAMI extends ICLA {
	public void getTrainingTestSet(Object[] keys, Instances instances, String positiveLabel, double percentileCutoff);

	public void getPredictedLabels(boolean suppress, Instances instances);

}
