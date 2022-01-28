package net.lifove.clami.factor;

import weka.core.Instances;

abstract class DataFactor {

	static String factorName ;
	static double value;
	
	abstract void computeValue(Instances instancesByCLA, Instances instances, double percentileCutoff, String positiveLabel); 
	
	abstract double getValue();
	
}
