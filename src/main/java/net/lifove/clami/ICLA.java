package net.lifove.clami;

import weka.core.Instances;

public interface ICLA {
	public void getResult(Instances instances, double percentileCutoff, String positiveLabel, boolean suppress, String fileName);

	public void getResult(Instances instances, double percentileCutoff, String positiveLabel, boolean suppress,
			boolean experimental, String fileName);

	public Instances clustering(Instances instances, double percentileCutoff, String positiveLabel);

	public void printResult(Instances instances, boolean experimental, String fileName, boolean suppress,
			String positiveLabel);

}
