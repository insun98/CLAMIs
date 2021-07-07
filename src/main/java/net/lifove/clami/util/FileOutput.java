package net.lifove.clami.util;

import java.io.FileWriter;
import java.io.IOException;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;

public class FileOutput {
	
	public void createFile() {
		
		try(CSVPrinter printer = new CSVPrinter(new FileWriter("result.csv"), CSVFormat.EXCEL)) {
			
		} catch(IOException ex) {
			ex.printStackTrace();
		}
	}

}
