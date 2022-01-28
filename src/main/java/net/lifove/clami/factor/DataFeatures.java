package net.lifove.clami.factor;

import java.util.ArrayList;

import weka.core.Instances;

public class DataFeatures {
	
	ArrayList<DataFactor> factors = new ArrayList<DataFactor>();

	public DataFeatures(Instances instancesByCLA, Instances instances, double percentileCutoff, String positiveLabel) {
		
		DataFactorGIR gir = new DataFactorGIR(instancesByCLA, instances, percentileCutoff, positiveLabel);
		factors.add(gir);
		
		double GIR = factors.get(0).getValue(); 
		
		System.out.println("GIR:" + GIR);
		
	}

}