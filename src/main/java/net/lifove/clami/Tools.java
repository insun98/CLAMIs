package net.lifove.clami;

import org.apache.commons.math3.stat.correlation.SpearmansCorrelation;
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;

import net.lifove.clami.util.Utils;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class Tools {
	CLA cla = new CLA();

	public double selectPercentileCutoff(Instances instances, String positiveLabel, Instances instancesByCLA,
			double score, double totalViolation, double sum) {
		double finalPercentileCutoff = 0.0;
		double percentileCutoff = 0.0;
		double maxScore = 0.0;

		for (percentileCutoff = 10.0; percentileCutoff < 100; percentileCutoff += 5) {
			maxScore = score;
			sum = 0.0;
			totalViolation = 0.0;
			double[] cutoffsForHigherValuesOfAttribute = Utils.getHigherValueCutoffs(instances, percentileCutoff);

			instancesByCLA = cla.clustering(instances, percentileCutoff, positiveLabel);

			HashMap<Integer, String> metricIdxWithTheSameViolationScores = Utils.getMetricIndicesWithTheViolationScores(
					instancesByCLA, cutoffsForHigherValuesOfAttribute, positiveLabel);
			Object[] keys = metricIdxWithTheSameViolationScores.keySet().toArray();

			for (Object key : keys) {
				String key1 = key.toString();
				totalViolation = totalViolation + Integer.parseInt(key1);
			}

			score = (sum / instances.numAttributes())
					+ (totalViolation / (instances.numAttributes() * instances.numInstances()));
			System.out.println("percentile: " + percentileCutoff + " score: " + score + " violation" + totalViolation);
			if (score < maxScore)
				finalPercentileCutoff = percentileCutoff;
		}
		percentileCutoff = finalPercentileCutoff;
		return percentileCutoff;
	}

	public void ksTest(Instances instancesByCLA, Instances instances, double percentileCutoff, String positiveLabel) {

		double criticalValue;
		double alpha = 1.224;
		int[] scoreOfInstances = new int[instances.numInstances()];
		int[] scoreOfInstancesCopy = new int[instances.numInstances()];
		int maxVote = 0;
		int count = 0;

		criticalValue = alpha * Math.sqrt((instances.numInstances() * 2) / Math.pow(instances.numInstances(), 2));

		for (int attrIdx = 0; attrIdx < instances.numAttributes() - 1; attrIdx++) {

			int k = 0;

			double[] X = instances.attributeToDoubleArray(attrIdx);
			Instances newInstances = new Instances(instances);

			for (int attrIdx1 = 0; attrIdx1 < instances.numAttributes() - 1; attrIdx1++) {

				double[] Y = instances.attributeToDoubleArray(attrIdx1);
				KolmogorovSmirnovTest ks = new KolmogorovSmirnovTest();
				double i = ks.kolmogorovSmirnovTest(X, Y);

				if (i < criticalValue) {
					newInstances.deleteAttributeAt(k);
				}

				if (i > criticalValue) {
					k++;
				}

			}

			if (newInstances.numAttributes() > 2) {
				instancesByCLA = cla.clustering(newInstances, percentileCutoff, positiveLabel);
				maxVote++;

				for (int instIdx = 0; instIdx < newInstances.numInstances(); instIdx++) {

					if (instancesByCLA.get(instIdx).value(newInstances.numAttributes() - 1) == Double
							.parseDouble(positiveLabel)) {
						scoreOfInstances[instIdx]++;
					}
				}
			}
		}

		scoreOfInstancesCopy = scoreOfInstances.clone();

		Arrays.sort(scoreOfInstances);
		int idx = (int) (instances.numInstances() * 0.25);
		int cutoffValue = scoreOfInstances[idx];

		for (int instIdx = 0; instIdx < instances.numInstances(); instIdx++) {

			if (scoreOfInstancesCopy[instIdx] >= cutoffValue)
				instancesByCLA.instance(instIdx).setClassValue(positiveLabel);
			else
				instancesByCLA.instance(instIdx).setClassValue(Utils.getNegLabel(instancesByCLA, positiveLabel));
		}
	}

	public void calculateSpearmanCorrelation(Instances instances, Instances instancesByCLA) {

		for (int attrIdx = 0; attrIdx < instances.numAttributes(); attrIdx++) {

			double avg = 0.0;
			double sum = 0.0;

			double[] X = instancesByCLA.attributeToDoubleArray(attrIdx);

			for (int i = 0; i < X.length; i++) {
				if (Double.isNaN(X[i])) {
					X[i] = 0;
				}
			}

			for (int attrIdx2 = 0; attrIdx2 < instances.numAttributes(); attrIdx2++) {
				if (attrIdx == attrIdx2)
					continue;

				double[] Y = instancesByCLA.attributeToDoubleArray(attrIdx2);

				for (int j = 0; j < Y.length; j++) {
					if (Double.isNaN(Y[j])) {
						Y[j] = 0;
					}
				}

				SpearmansCorrelation correlation1 = new SpearmansCorrelation();

				double correlation = correlation1.correlation(X, Y);

				if (Double.isNaN(correlation))
					correlation = 0;
				sum = sum + correlation;

				avg = sum;
			}

			avg = avg / (instances.numAttributes() - 1);

			System.out.println(attrIdx + " " + avg);

		}
	}

}
