package net.lifove.clami.util;

import net.lifove.clami.CLA;
import net.lifove.clami.CLAMI;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSink;

public class GenerateFinalWithPrediction {
	
	/**
	 * This method is for generate arff file for result. 
	 * The label of this result file is predicted wih clami or other version that come though the parameter.
	 * @param version String value of version (ex. "CLA" or "CLAMI" ...)
	 * @param instances 
	 * @param percentileCutoff
	 * @param positiveLabel
	 * @param filePath
	 */
	public void generateResult(String version, Instances instances, double percentileCutoff, String positiveLabel, String filePath) {
		
		String fileName ;
		
		// version of cla 
		if (version.equals("CLA")) {
			
			fileName = "CLA-label-" + extractOnlyFileName(filePath);
			
			CLA cla = new CLA();
			Instances instancesByCLA = cla.clustering(instances, percentileCutoff, positiveLabel); 
			
			try {
				DataSink.write(fileName, instancesByCLA);
			} catch (Exception e) {
				   System.err.println("Failed to save data to: " + fileName);
				   e.printStackTrace();
			}
		}
		// version of clami 
		
	}
	
	/**
	 * To return only the file name
	 * @param file
	 * @return
	 */
	public String extractOnlyFileName(String file) {
		
		String[] name;
		name = file.split("/"); 
	
		return name[name.length-1];
	}

}

