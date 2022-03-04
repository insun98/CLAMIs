package net.lifove.clami;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.math3.stat.correlation.SpearmansCorrelation;

import net.lifove.clami.factor.DataFactor;
import net.lifove.clami.factor.DataFactorGIR;
import net.lifove.clami.factor.DataFeasibilityChecker;
import net.lifove.clami.percentileselectors.IPercentileSelector;
import net.lifove.clami.percentileselectors.PercentileSelectorBottom;
import net.lifove.clami.percentileselectors.PercentileSelectorTop;
import net.lifove.clami.util.Utils;
import weka.core.Instances;

/**
 * CLAMI implementation: CLAMI: Defect Prediction on Unlabeled Datasets, in
 * Proceedings of the 30th IEEE/ACM International Conference on Automated
 * Software Engineering (ASE 2015), Lincoln, Nebraska, USA, November 9 - 13,
 * 2015
 * 
 * @author JC
 *
 */
public class Main implements IPercentileSelector{

	String dataFilePath;
	String labelName;
	String posLabelValue;
	double percentileCutoff = 50;
	double factorCutoff = 0.2;
	String version="";
	//boolean isClami="";
	//boolean isClabi="";
	boolean help = false;
	boolean suppress = false;
	String experimental;
	String mlAlg = "";
	String percentileOption;
	String factorCutoffOption;
	int sort = 0;

	
	public static void main(String[] args) throws FileNotFoundException, IOException {

		new Main().runner(args);

	}

	void runner(String[] args) throws FileNotFoundException, IOException {
		
		Options options = createOptions();
	
		if (parseOptions(options, args)) {
			handleAuxOptions(options);

			File dir = new File(dataFilePath);

			if (dir.isDirectory()) {
				processMultipleFileInOneDirectory(dir);
			} else {
				processSingleFile();
			}
		}
			
		if (version.equals("CLAMI"))
			Utils.makeFile("CLAMI");
		
		else if (version.equals("CLABI")) 
			Utils.makeFile("CLABI");
		
		else if (version.equals("CLAMI+"))
			Utils.makeFile("CLAMI+");
		
		else if (version.equals("CLABI+"))
			Utils.makeFile("CLABI+");
		
		else if(version.equals("CLA+"))
			Utils.makeFile("CLA+");
		
		else if(version.equals("CLA"))
			Utils.makeFile("CLA");
	}

	private void processSingleFile() throws FileNotFoundException, IOException {
		// load an arff file
		Instances instances = Utils.loadArff(dataFilePath, labelName);
		
		percentileCutoff = getOptimalPercentile(instances, posLabelValue, percentileOption);

		if (instances != null) {
			double unit = (double) 100 / (instances.numInstances());
			// double unitFloor = Math.floor(unit);
			double unitCeil = Math.ceil(unit);

			// TODO need to check how median is computed
			if (unit >= 1 && 100 - unitCeil < percentileCutoff) {
				System.err.println("Cutoff percentile must be 0 < and <=" + (100 - unitCeil));
				return;
			}

			// For computing data factors 
			
			DataFeasibilityChecker data = new DataFeasibilityChecker();
			data.computeNumberOfGroups(instances, instances, percentileCutoff, posLabelValue);
			DataFactor GIR = data.getFactors("GIR");
			DataFactor GMR = data.getFactors("GMR");
			
			double finalFactor = 3 * GIR.getValue() + GMR.getValue();
			
			if(finalFactor >= factorCutoff)
			{
				System.out.println(finalFactor + " 해당 파일은 CLA/CLAMI에 적합한 데이터 입니다.");
				if (experimental == null || experimental.equals("")) {
						// do prediction
						prediction(instances, posLabelValue, false, dataFilePath);
					} else {
						experiment(instances, posLabelValue, dataFilePath);
					}
			}

			else
				System.out.println(finalFactor +" 해당 파일은 CLA/CLAMI에 적합하지 않은 데이터 입니다.");
		}
	}

	private void processMultipleFileInOneDirectory(File dir) throws FileNotFoundException, IOException {
		File[] fileList = dir.listFiles();

		for (File file : fileList) {
			// load an arff file
			Instances instances = Utils.loadArff(file.toString(), labelName);
			if (instances == null) continue ;
			
			//percentileCutoff = getOptimalPercentile(instances, posLabelValue, percentileOption);

			if (instances != null) {
				double unit = (double) 100 / (instances.numInstances());
				// double unitFloor = Math.floor(unit);
				double unitCeil = Math.ceil(unit);

				// TODO need to check how median is computed
				if (unit >= 1 && 100 - unitCeil < percentileCutoff) {
					System.err.println("Cutoff percentile must be 0 < and <=" + (100 - unitCeil));
					return;
				}
				
				// For computing data factors 
				DataFeasibilityChecker data = new DataFeasibilityChecker();
				data.computeNumberOfGroups(instances, instances, percentileCutoff, posLabelValue);

				if (experimental == null || experimental.equals("")) {
					// do prediction
					prediction(instances, posLabelValue, false, file.toString());
				} else {
					experiment(instances, posLabelValue, file.toString());
				}
			}
			
		}
	}

	private void handleAuxOptions(Options options) {
		if (help) {
			printHelp(options);
			return;
		}
		
		if(!version.equals("CLAMI") && !version.equals("CLAMI+") && !version.equals("CLABI") && !version.equals("CLABI+") && !version.equals("CLA") && !version.equals("CLA+")) {
			System.err.println("Version format is incorrect. Check your version option.");
			return;
		}

		// exit when percentile range is not correct (it should be 0 < range <= 100)
		if (percentileCutoff <= 0 || 100 < percentileCutoff) {
			System.err.println("Cutoff percentile must be 0 < and <=100");
			return;
		}

		// exit experimental option format is not correct
		if (experimental != null && !checkExperimentalOption(experimental)) {
			System.err.println(
					"Experimental option format is incorrect. Option format: [# of folds]:[# of repetition]. "
							+ "E.g, -e 2:500 (Two-fold cross validation 500 repetition");
			return;
		}
	}

	private boolean checkExperimentalOption(String expOpt) {
		Pattern pattern = Pattern.compile("^[0-9]+:[0-9]");
		Matcher m = pattern.matcher(expOpt);
		return m.find();
	}

	private void experiment(Instances instances, String posLabelValue, String fileName)
			throws FileNotFoundException, IOException {

		String[] splitOptions = experimental.split(":");
		int folds = Integer.parseInt(splitOptions[0]);
		int numRuns = Integer.parseInt(splitOptions[1]);

		String source = dataFilePath.substring(dataFilePath.lastIndexOf(File.separator) + 1).replace(".arff", "");

		for (int repeat = 0; repeat < numRuns; repeat++) {

			// randomize with different seed for each iteration
			instances.randomize(new Random(repeat));
			instances.stratify(folds);

			for (int fold = 0; fold < folds; fold++) {
				System.out.print(repeat + "," + fold + "," + source + ",");
				Instances targetInstances = instances.testCV(folds, fold);
				prediction(targetInstances, posLabelValue, true, fileName);
				System.out.println();
			}
		}
	}

	void prediction(Instances instances, String positiveLabel, boolean isExperimental, String fileName)
			throws FileNotFoundException, IOException {

		ICLA claApproach;
		ICLAMI clamiApproach;
		
		if (version.equals("CLABI")) {
			clamiApproach = new CLABI(mlAlg,isExperimental);
			clamiApproach.getResult(instances, percentileCutoff, positiveLabel, suppress, fileName);
		}
		else if (version.equals("CLAMI")) {
			clamiApproach = new CLAMI(mlAlg,isExperimental);
			clamiApproach.getResult(instances, percentileCutoff, positiveLabel, suppress, fileName);
		}
		else if (version.equals("CLABI+")) {
			clamiApproach = new CLABIPlus(mlAlg,isExperimental);
			clamiApproach.getResult(instances, percentileCutoff, positiveLabel, suppress, fileName);
		}
		else if (version.equals("CLAMI+")) {
			clamiApproach = new CLAMIPlus(mlAlg,isExperimental);
			clamiApproach.getResult(instances, percentileCutoff, positiveLabel, suppress, fileName);
		}
		else if (version.equals("CLA+")) {
			claApproach = new CLAPlus();
			claApproach.getResult(instances, percentileCutoff, positiveLabel, suppress, fileName);
		}
		else  {
			claApproach = new CLA();
			claApproach.getResult(instances, percentileCutoff, positiveLabel, suppress, fileName);
		}
		
	} 

	private void printHelp(Options options) {
		// automatically generate the help statement
		HelpFormatter formatter = new HelpFormatter();
		String header = "Execute CLA/CLAMI unsuprvised defect predicition. On Windows, use CLAMI.bat instead of ./CLAMI";
		String footer = "\nPlease report issues at https://github.com/lifove/CLAMI/issues";
		formatter.printHelp("./CLAMI", header, options, footer, true);
	}

	Options createOptions() {

		// create Options object
		Options options = new Options();

		// add options
		options.addOption(Option.builder("f").longOpt("file").desc("Arff file path to predict defects").hasArg()
				.argName("file").required().build());

		options.addOption(Option.builder("h").longOpt("help").desc("Help").build());

		options.addOption(Option.builder("c").longOpt("cutoff")
				.desc("Options for selecting percentilecutoff of the same values, "+
				"t for top percentile cutoff, "+
				"b for bottom percentile cutoff, "+
				"m for median (50).").hasArg()
				.argName("cutoff percentile").required().build());

		options.addOption(Option.builder("s").longOpt("suppress")
				.desc("Suppress detailed prediction results. Only works when the arff data is labeled.").build());

		options.addOption(Option.builder("l").longOpt("lable").desc("Label (Class attrubite) name").hasArg()
				.argName("attribute name").required().build());

		options.addOption(Option.builder("p").longOpt("poslabel").desc(
				"String value of buggy label. Since CLA/CLAMI works for unlabeld data (in case of weka arff files, labeled as '?',"
						+ " it is not necessary to use this option. " + "However, if the data file is labeled, "
						+ "it will show prediction results in terms of precision, recall, and f-measure for evaluation puerpose.")
				.hasArg().required().argName("postive label value").build());
		
		options.addOption(Option.builder("v").longOpt("version").desc(
				"Options for selecting a version of the program. Insert  CLA for CLA, "+ 
				"CLAMI for CLAMI, "+
				"CLABI for CLABI. " +
				"(Add + at the end to run plus version. Ex) CLA+, CLAMI+, CLABI+)")
				.hasArg()
				.required()
				.argName("version of the program").build());

		options.addOption(Option.builder("e").longOpt("experimental").desc(
				"Options for experimenets to compare CLA/CLAMI with other cross-project defect prediction approaches by k-fold cross validation. "
						+ "Support k-fold cross validation n times. "
						+ "Option format: [# of folds]:[# of repetition]. E.g, -e 2:500 (Two-fold cross validation 500 repetition")
				.hasArg().argName("#folds:#repeat").build());

		options.addOption(Option.builder("a").longOpt("mlalgorithm")
				.desc("Specify weka classifier (Default: weka.classifiers.functions.Logistic)").hasArg()
				.argName("Algorithm").build());
		
		options.addOption(Option.builder("fc").longOpt("factorCutoff")
				.desc("Options for selecting cutoff value of factor (Default: 0.2)").hasArg()
				.argName("factor cutoff").build());
		
		return options;

	} 

	boolean parseOptions(Options options, String[] args) {

		CommandLineParser parser = new DefaultParser();

		try {

			CommandLine cmd = parser.parse(options, args);

			dataFilePath = cmd.getOptionValue("f");
			labelName = cmd.getOptionValue("l");
			posLabelValue = cmd.getOptionValue("p");
			if (cmd.getOptionValue("c") != null)
				percentileOption = cmd.getOptionValue("c");
			if(cmd.getOptionValue("v")!=null)
				version = cmd.getOptionValue("v"); 
			help = cmd.hasOption("h");
			suppress = cmd.hasOption("s");
			experimental = cmd.getOptionValue("e");
			mlAlg = cmd.getOptionValue("a");
			if(cmd.getOptionValue("fc")!=null)
			{
				factorCutoffOption = cmd.getOptionValue("fc");
				factorCutoff = Double.parseDouble(factorCutoffOption);	
			}

		} catch (Exception e) {
			printHelp(options);
			return false;
		}
 
		return true;
	}
	
	public double getOptimalPercentile(Instances instances, String positiveLabel, String percentileOption){
		
		double percentileCutoff;
		double[] instancesValue;
		double[] instancesClassvalue = new double[instances.numInstances()];
		Instances instancesByCLA;
		HashMap<Double, Integer> percentileCorrelation = new HashMap<>();
		CLA cla = new CLA();
		int numOfCorrelation = 0;
		
		for(percentileCutoff = 1.0; percentileCutoff<100; percentileCutoff+=1){
			
			instancesByCLA = cla.clustering(instances, percentileCutoff, positiveLabel);
			//System.out.println("Percentile" + percentileCutoff);
			for (int instIdx = 0; instIdx < instances.numInstances(); instIdx++) 
			{
				if (Utils.getStringValueOfInstanceLabel(instancesByCLA, instIdx).equals(positiveLabel)) {
					instancesClassvalue[instIdx] = 1.0;
				} else {
					instancesClassvalue[instIdx] = 0.0;
				}
			}
			
			for (int attrIdx = 0; attrIdx < instances.numAttributes(); attrIdx++) {
				
				instancesValue = instancesByCLA.attributeToDoubleArray(attrIdx);
				//System.out.println("instancesValue" + instancesValue[attrIdx]);
				 
				for(int i =0; i<instancesValue.length; i++) {
					if(Double.isNaN(instancesValue[i])) {
						instancesValue[i]=0;
					}
				}
				
				SpearmansCorrelation correlation1 = new SpearmansCorrelation();
					
				double correlation = correlation1.correlation(instancesValue, instancesClassvalue);
				if(Double.isNaN(correlation))
					correlation = 0;
						
				//System.out.println("metric num: " + (attrIdx+1) + " "+ correlation);
					
				if(correlation > 0.5)
					numOfCorrelation++;
			}
			
			percentileCorrelation.put(percentileCutoff, numOfCorrelation);
			numOfCorrelation = 0;
		}
		
		if(percentileOption.equals("t"))
			percentileCutoff = PercentileSelectorTop.getTopPercentileCutoff(instances,positiveLabel, percentileCorrelation);
		
		else if(percentileOption.equals("b"))
			percentileCutoff = PercentileSelectorBottom.getBottomPercentileCutoff(instances,positiveLabel,percentileCorrelation);
		
		else if(percentileOption.equals("m"))
			percentileCutoff = 50;
		
		
		
		return percentileCutoff;
		
	}

@Override
public double getTopPercentileCutoff(Instances instances, String positiveLabel) {
	// TODO Auto-generated method stub
	return 0;
}

@Override
public double getBottomPercentileCutoff(Instances instances, String positiveLabel) {
	// TODO Auto-generated method stub
	return 0;
}
}
