package net.lifove.clami.percentileselectors;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import org.apache.commons.math3.stat.correlation.SpearmansCorrelation;

import weka.core.Instances;

public class PercentileSelectorBottom {
	
public static double getBottomPercentileCutoff(Instances instances, String positiveLabel,  HashMap<Double, Integer> percentileCorrelation){
		
		
		List<Map.Entry<Double, Integer>> entryList = new LinkedList<>(percentileCorrelation.entrySet());
		entryList.sort(Map.Entry.comparingByValue());
		/*
		for(Map.Entry<Double, Integer> entry : entryList){
		    System.out.println("key : " + entry.getKey() + ", value : " + entry.getValue());	    
		}
		*/
		int max = entryList.get(entryList.size()-1).getValue();
		Vector<Double> percentileList = new Vector<Double>();
		
		for (double key : percentileCorrelation.keySet()) {
			
            if (max == percentileCorrelation.get(key)) {
            	
            	percentileList.add(key);     
            }
        }
		Object[] percentileListArray = percentileList.toArray();
		Arrays.sort(percentileListArray);
		
		for(Object array : percentileListArray)
		{
			System.out.println(array +" ");
		}
		
		System.out.println("Bottom percentile: " + (double) percentileListArray[0]);
		
		return (double) percentileListArray[0];
		
	}

}
