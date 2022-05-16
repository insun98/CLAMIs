package net.lifove.clami.factor;

import java.util.ArrayList;

/**
 * This class is for data factor 'GIR'
 */
public class DataFactorGIR extends DataFactor {

	private ArrayList<DataFactor> factors;
	private double numOfInstances;
	private double numOfGroups;
	
	public DataFactorGIR(ArrayList<DataFactor> factors) {

		this.factors = factors;
		this.factorName = "GIR";
		
	}
	
	@Override
	public DataFactor computeValue() {
		
		for (DataFactor i: factors) {
			if (i.factorName.equals("numberOfInstances")) {
				numOfInstances = i.getValue();
			}
			if (i.factorName.equals("numberOfGroups")) {
				numOfGroups = i.getValue(); 
			}
		}
		
		this.value = (double) numOfGroups / numOfInstances ;
		
		return this;
		
	}

}
