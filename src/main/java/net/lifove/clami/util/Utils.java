package net.lifove.clami.util;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import org.apache.commons.math3.stat.StatUtils;
import com.google.common.primitives.Doubles;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.RemoveRange;
import org.apache.poi.xssf.usermodel.XSSFCell;
import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

public class Utils {
	
	private static ArrayList<ArrayList<Object>> Data = new ArrayList<ArrayList<Object>>();

	private static int number = 0;

	/**
	 * To create a result file 
	 * @param versionName
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public static void makeFile(String versionName) throws FileNotFoundException, IOException {
		XSSFWorkbook workbook = new XSSFWorkbook();
		XSSFSheet sheet = workbook.createSheet("Result");
		XSSFRow row = null;
		XSSFCell cell = null;

		Object[] header = { "File Name", "TP", "FP", "TN", "FN", "Precision", "Recall", "F-Measure", "AUC", "MCC" };

		row = sheet.createRow(0);
		int columnCount = 0;
		for (Object field : header) {
			cell = row.createCell(columnCount++);
			cell.setCellValue((String) field);
		}
		for (int i = 1; i <= Data.size(); i++) {
			ArrayList<Object> arrData = Data.get(i-1);
			row = sheet.createRow(i);
			for (int k = 0; k < arrData.size(); k++) {
				sheet.autoSizeColumn(k);
				cell = row.createCell(k);
				if (arrData.get(k) instanceof String) 
					cell.setCellValue((String) arrData.get(k));
				else if (arrData.get(k) instanceof Integer) 
					cell.setCellValue((Integer) arrData.get(k));
				else 
					cell.setCellValue((Double) arrData.get(k));
			}
		}
		try {
			String strFilePath = versionName + "_Result.xlsx";
			FileOutputStream fOut = new FileOutputStream(strFilePath);
			workbook.write(fOut);
			workbook.close();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(0);
		}

	}


	/**
	 * Print prediction performance in terms of TP, TN, FP, FN, precision, recall, and f1. 
	 * @param tP
	 * @param tN
	 * @param fP
	 * @param fN
	 * @throws IOException
	 * @throws FileNotFoundException
	 */
	public static void printEvaluationResultCLA(int TP,int  TN, int FP, int FN, boolean experimental, String fileName) {

		double precision = (double) TP / (TP + FP);
		double recall = (double)TP / (TP + FN);
		double f1 = (2 * (precision * recall)) / (precision + recall);
		if (!experimental) {
			System.out.println(fileName);
			System.out.println("TP: " + TP);
			System.out.println("FP: " + FP);
			System.out.println("TN: " + TN);
			System.out.println("FN: " + FN);

			System.out.println("precision: " + precision);
			System.out.println("recall: " + recall);
			System.out.println("f1: " + f1);
		} else
			System.out.print(precision + "," + recall + "," + f1);


		ArrayList<Object> subData = new ArrayList<Object>();

		subData.add(0, fileName);
		subData.add(1, TP);
		subData.add(2, FP);
		subData.add(3, TN);
		subData.add(4, FN);
		subData.add(5, precision);
		subData.add(6, recall);
		subData.add(7, f1);

		Data.add(number, subData);

		number++;

	}

	/**
	 * Print prediction performance in terms of TP, TN, FP, FN, precision, recall, f1, AUC, and MCC. 
	 * @param instances
	 * @param testInstances
	 * @param trainingInstances
	 * @param classifier
	 * @param positiveLabel
	 * @param experimental
	 * @param fileName
	 */
	public static void printEvaluationResult(Instances instances,Instances testInstances, Instances trainingInstances,Classifier classifier, String positiveLabel, boolean experimental, String fileName) {

		Evaluation eval = null;
		try {
			eval = new Evaluation(trainingInstances);
			eval.evaluateModel(classifier, testInstances);
		} catch (Exception e1) {
			e1.printStackTrace();
		}

		double TP = eval.truePositiveRate(instances.classAttribute().indexOfValue(positiveLabel));
		double FP = eval.falsePositiveRate(instances.classAttribute().indexOfValue(positiveLabel));
		double TN = eval.precision(instances.classAttribute().indexOfValue(positiveLabel));
		double FN = eval.precision(instances.classAttribute().indexOfValue(positiveLabel));

		double precision = eval.precision(instances.classAttribute().indexOfValue(positiveLabel));
		double recall = eval.recall(instances.classAttribute().indexOfValue(positiveLabel));
		double f1 = eval.fMeasure(instances.classAttribute().indexOfValue(positiveLabel));
		double AUC = eval.areaUnderROC(instances.classAttribute().indexOfValue(positiveLabel));
		double MCC = eval.matthewsCorrelationCoefficient(instances.classAttribute().indexOfValue(positiveLabel));

		if (!experimental) {
			System.out.println(fileName);
			System.out.println("TP: " + TP);
			System.out.println("FP: " + FP);
			System.out.println("TN: " + TN);
			System.out.println("FN: " + FN);


			System.out.println("precision: " + precision);
			System.out.println("recall: " + recall);
			System.out.println("f1: " + f1);

			System.out.println("AUC: " + AUC);
			System.out.println("MCC: " + MCC);

		} else 
			System.out.print(precision + "," + recall + "," + f1 + "," + AUC + "," + MCC);


		ArrayList<Object> subData = new ArrayList<Object>();

		subData.add(0, fileName);
		subData.add(1, TP);
		subData.add(2, FP);
		subData.add(3, TN);
		subData.add(4, FN);
		subData.add(5, precision);
		subData.add(6, recall);
		subData.add(7, f1);
		subData.add(8, AUC);
		subData.add(9, MCC);

		Data.add(number, subData);

		number++;

	}

	/**
	 * Get higher value cutoffs for each attribute
	 * 
	 * @param instances
	 * @param percentileCutoff
	 * @return double[]
	 */
	public static double[] getHigherValueCutoffs(Instances instances, double percentileCutoff) {
		// compute median values for attributes
		double[] cutoffForHigherValuesOfAttribute = new double[instances.numAttributes()];

		for (int attrIdx = 0; attrIdx < instances.numAttributes(); attrIdx++) {
			if (attrIdx == instances.classIndex())
				continue;
			cutoffForHigherValuesOfAttribute[attrIdx] = StatUtils.percentile(instances.attributeToDoubleArray(attrIdx),
					percentileCutoff);
		}
		return cutoffForHigherValuesOfAttribute;
	}

	/**
	 * Return the HashMap that the key is Metric Violation Score and the value is
	 * string of metric indexes
	 * 
	 * @param instances
	 * @param cutoffsForHigherValuesOfAttribute
	 * @param positiveLabel
	 * @return HashMap<Integer, String>
	 */
	public static HashMap<Integer, String> getMetricIndicesWithTheViolationScores(Instances instances,
			double[] cutoffsForHigherValuesOfAttribute, String positiveLabel) {

		int[] violations = new int[instances.numAttributes()];

		for (int attrIdx = 0; attrIdx < instances.numAttributes(); attrIdx++) {
			if (attrIdx == instances.classIndex()) {
				violations[attrIdx] = instances.numInstances(); // make this as max to ignore since our concern is
				// minimum violation.
				continue;
			}

			for (int instIdx = 0; instIdx < instances.numInstances(); instIdx++) {
				if (instances.get(instIdx).value(attrIdx) <= cutoffsForHigherValuesOfAttribute[attrIdx] && instances
						.get(instIdx).classValue() == instances.classAttribute().indexOfValue(positiveLabel)) {
					violations[attrIdx]++;
				} else if (instances.get(instIdx).value(attrIdx) > cutoffsForHigherValuesOfAttribute[attrIdx]
						&& instances.get(instIdx).classValue() == instances.classAttribute()
						.indexOfValue(getNegLabel(instances, positiveLabel))) {
					violations[attrIdx]++;
				}
			}
		}

		HashMap<Integer, String> metricIndicesWithTheSameViolationScores = new HashMap<Integer, String>();

		for (int attrIdx = 0; attrIdx < instances.numAttributes(); attrIdx++) {
			if (attrIdx == instances.classIndex()) {
				continue;
			}

			int key = violations[attrIdx];

			if (!metricIndicesWithTheSameViolationScores.containsKey(key)) {
				metricIndicesWithTheSameViolationScores.put(key, (attrIdx + 1) + ",");
			} else {
				String indices = metricIndicesWithTheSameViolationScores.get(key) + (attrIdx + 1) + ",";
				metricIndicesWithTheSameViolationScores.put(key, indices);
			}
		}

		return metricIndicesWithTheSameViolationScores;
	}

	/**
	 * Get the selected instance for the instance selection
	 * 
	 * @param instances
	 * @param cutoffsForHigherValuesOfAttribute
	 * @param positiveLabel
	 * @return String
	 */
	public static String getSelectedInstances(Instances instances, double[] cutoffsForHigherValuesOfAttribute,
			String positiveLabel) {

		int[] violations = new int[instances.numInstances()];

		for (int instIdx = 0; instIdx < instances.numInstances(); instIdx++) {

			for (int attrIdx = 0; attrIdx < instances.numAttributes(); attrIdx++) {
				if (attrIdx == instances.classIndex())
					continue; // no need to compute violation score for the class attribute

				if (instances.get(instIdx).value(attrIdx) <= cutoffsForHigherValuesOfAttribute[attrIdx] && instances
						.get(instIdx).classValue() == instances.classAttribute().indexOfValue(positiveLabel)) {
					violations[instIdx]++;
				} else if (instances.get(instIdx).value(attrIdx) > cutoffsForHigherValuesOfAttribute[attrIdx]
						&& instances.get(instIdx).classValue() == instances.classAttribute()
						.indexOfValue(getNegLabel(instances, positiveLabel))) {
					violations[instIdx]++;
				}
			}
		}

		String selectedInstances = "";

		for (int instIdx = 0; instIdx < instances.numInstances(); instIdx++) {
			if (violations[instIdx] > 0)
				selectedInstances += (instIdx + 1) + ","; // let the start attribute index be 1
		}

		return selectedInstances;
	}

	/**
	 * Get the negative label string value from the positive label value
	 * 
	 * @param instances
	 * @param positiveLabel
	 * @return String
	 */
	static public String getNegLabel(Instances instances, String positiveLabel) {
		if (instances.classAttribute().numValues() == 2) {
			int posIndex = instances.classAttribute().indexOfValue(positiveLabel);
			if (posIndex == 0)
				return instances.classAttribute().value(1);
			else
				return instances.classAttribute().value(0);
		} else {
			System.err.println("Class labels must be binary");
			System.exit(0);
		}
		return null;
	}

	/**
	 * Load Instances from arff file. Last attribute will be set as class attribute
	 * 
	 * @param path arff file path
	 * @return Instances
	 */
	public static Instances loadArff(String path, String classAttributeName) {
		Instances instances = null;
		BufferedReader reader;
		try {
			reader = new BufferedReader(new FileReader(path));
			instances = new Instances(reader);
			reader.close();
			instances.setClassIndex(instances.attribute(classAttributeName).index());
		} catch (NullPointerException e) {
			System.err.println("Class label name, " + classAttributeName
					+ ", does not exist! Please, check if the label name is correct.");
			instances = null;
		} catch (FileNotFoundException e) {
			System.err.println("Data file, " + path + ", does not exist. Please, check the path again!");
		} catch (IOException e) {
			System.err.println("I/O error! Please, try again!");
		}

		return instances;
	}

	/**
	 * Get label value of an instance
	 * 
	 * @param instances
	 * @param instance  index
	 * @return string label of an instance
	 */
	static public String getStringValueOfInstanceLabel(Instances instances, int intanceIndex) {
		return instances.instance(intanceIndex).stringValue(instances.classIndex());
	}

	/**
	 * Get median from ArraList<Double>
	 * 
	 * @param values
	 * @return double
	 */
	static public double getMedian(ArrayList<Double> values) {
		return getPercentile(values, 50);
	}

	/**
	 * Get a value in a specific percentile from ArraList<Double>
	 * 
	 * @param values
	 * @return double
	 */
	static public double getPercentile(ArrayList<Double> values, double percentile) {
		return StatUtils.percentile(getDoublePrimitive(values), percentile);
	}

	/**
	 * Get primitive double form ArrayList<Double>
	 * 
	 * @param values
	 * @return double[]
	 */
	public static double[] getDoublePrimitive(ArrayList<Double> values) {
		return Doubles.toArray(values);
	}

	/**
	 * Get instances by removing specific attributes
	 * 
	 * @param instances
	 * @param attributeIndices attribute indices (e.g., 1,3,4) first index is 1
	 * @param invertSelection  for invert selection, if true, select attributes with
	 *                         attributeIndices bug if false, remote attributes with
	 *                         attributeIndices
	 * @return new instances with specific attributes
	 */
	static public Instances getInstancesByRemovingSpecificAttributes(Instances instances, String attributeIndices,
			boolean invertSelection) {
		Instances newInstances = new Instances(instances);

		Remove remove;

		remove = new Remove();
		remove.setAttributeIndices(attributeIndices);
		remove.setInvertSelection(invertSelection);
		try {
			remove.setInputFormat(newInstances);
			newInstances = Filter.useFilter(newInstances, remove);
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(0);
		}

		return newInstances;
	}

	/**
	 * Get instances by removing specific instances
	 * 
	 * @param instances
	 * @param instance  indices (e.g., 1,3,4) first index is 1
	 * @param option    for invert selection
	 * @return selected instances
	 */
	static public Instances getInstancesByRemovingSpecificInstances(Instances instances, String instanceIndices,
			boolean invertSelection) {
		Instances newInstances = null;

		RemoveRange instFilter = new RemoveRange();
		instFilter.setInstancesIndices(instanceIndices);
		instFilter.setInvertSelection(invertSelection);

		try {
			instFilter.setInputFormat(instances);
			newInstances = Filter.useFilter(instances, instFilter);
		} catch (Exception e) {
			e.printStackTrace();
		}

		return newInstances;
	}

}
