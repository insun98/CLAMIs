package net.lifove.clami.factor;

import java.util.ArrayList;

public class DataFactorGMR extends DataFactor {

	private ArrayList<DataFactor> factors;
	private double numOfMetrics;
	private double numOfGroups;
	
	public DataFactorGMR(ArrayList<DataFactor> factors) {

		this.factors = factors;
		this.factorName = "GMR";
		
	}
	
	@Override
	public DataFactor computeValue() {
		
		for (DataFactor i: factors) {
			if (i.factorName.equals("numberOfMetrics")) {
				numOfMetrics = i.getValue();
			}
			if (i.factorName.equals("numberOfGroups")) {
				numOfGroups = i.getValue(); 
			}
		}
		
		this.value = (double) numOfGroups / numOfMetrics ;
		
		return this;
		
	}

}
