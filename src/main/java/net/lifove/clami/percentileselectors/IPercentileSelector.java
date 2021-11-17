package net.lifove.clami.percentileselectors;



import weka.core.Instances;

public interface IPercentileSelector {
	
	public double getTopPercentileCutoff(Instances instances, String positiveLabel);
	 
	public double getBottomPercentileCutoff(Instances instances, String positiveLabel);

}
