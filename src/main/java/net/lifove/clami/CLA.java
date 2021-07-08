package net.lifove.clami;

import net.lifove.clami.util.Utils;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class CLA {
	/**
	 * Get CLA result
	 * @param instances
	 * @param percentileCutoff: cutoff percentile for top and bottom clusters
	 * @param positiveLabel positive label string value
	 * @param suppress detailed prediction results
	 * @return instances labeled by CLA
	 */
	public static void getCLAResult(Instances instances,double percentileCutoff,String positiveLabel,boolean suppress, boolean isDegree) {
		getCLAResult(instances,percentileCutoff,positiveLabel,suppress,false, isDegree); // no experimental as default
	}
	
	/**
	 * Get CLA result
	 * @param instances
	 * @param percentileCutoff cutoff percentile for top and bottom clusters
	 * @param positiveLabel positive label string value
	 * @param suppress detailed prediction results
	 * @param experimental option to display a result in a line;
	 * @return instances labeled by CLA
	 */
	public static void getCLAResult(Instances instances,double percentileCutoff,String positiveLabel,boolean suppress, boolean experimental, boolean isDegree) {
		Instances instancesByCLA = Utils.getInstancesByCLA(instances, percentileCutoff, positiveLabel, isDegree);
		
		// Print CLA results
		int TP=0, FP=0,TN=0, FN=0;
		
		for(int instIdx = 0; instIdx < instancesByCLA.numInstances(); instIdx++){
			if(!suppress)
				System.out.println("CLA: Instance " + (instIdx+1) + " predicted as, " + Utils.getStringValueOfInstanceLabel(instancesByCLA,instIdx) +
						", (Actual class: " + Utils.getStringValueOfInstanceLabel(instances,instIdx) + ") ");
			
			// compute T/F/P/N for the original instances labeled.
			if(!Double.isNaN(instances.get(instIdx).classValue())){
				if(Utils.getStringValueOfInstanceLabel(instancesByCLA,instIdx).equals(Utils.getStringValueOfInstanceLabel(instances,instIdx))){
					if(Utils.getStringValueOfInstanceLabel(instancesByCLA,instIdx).equals(positiveLabel))
						TP++;
					else
						TN++;
				}else{
					if(Utils.getStringValueOfInstanceLabel(instancesByCLA,instIdx).equals(positiveLabel))
						FP++;
					else
						FN++;
				}
			}
		}
		
		if (TP+TN+FP+FN>0)
			Utils.printEvaluationResultCLA(TP, TN, FP, FN, experimental);
		else if(suppress)
			System.out.println("No labeled instances in the arff file. To see detailed prediction results, try again without the suppress option  (-s,--suppress)");
	}


}
