package net.lifove.clami;

import weka.core.Instances;

public class CLABI {
	
		/**
		 * Get CLAMI result. Since CLAMI is the later steps of CLA, to get instancesByCLA use getCLAResult.
		 * @param testInstances
		 * @param instancesByCLA
		 * @param positiveLabel
		 */
		public static void getCLABIResult(Instances testInstances, Instances instances, String positiveLabel,double percentileCutoff,boolean suppress,String mlAlg, boolean isDegree, int sort) {
			getCLABIResult(testInstances,instances,positiveLabel,percentileCutoff,suppress,false,mlAlg, isDegree, sort); //no experimental as default
		
	}
		/**
		 * Get CLAMI result. Since CLAMI is the later steps of CLA, to get instancesByCLA use getCLAResult.
		 * @param testInstances
		 * @param instancesByCLA
		 * @param positiveLabel
		 */
		public static void getCLABIResult(Instances testInstances, Instances instances, String positiveLabel,double percentileCutoff, boolean suppress, boolean experimental, String mlAlg, boolean isDegree, int sort) {
		
		}
}
