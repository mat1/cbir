import java.util.Map;
import java.util.Vector;


public final class NaiveBayesClassifier implements IClassifier {

	private final int classCnt;
	private final int featureCnt;
	
	private final String[] classNames;
	private final float[] aPriori;
	private final int[] documentCounts;
	private final int[] totalFeatureCounts;
	private final int[][] featureCounts;
	private final double[][] probabilities;
	
	private int totalDocuments = 0;
	
	public NaiveBayesClassifier(int classes, int K) {
		this.featureCnt = K;
		this.classCnt = classes;
		this.classNames = new String[classes];
		this.aPriori = new float[classes];
		this.documentCounts = new int[classes];
		this.totalFeatureCounts = new int [classes];
		this.featureCounts = new int [classes][featureCnt];
		this.probabilities = new double[classes][featureCnt];
	}
	
	@Override
	public void learn(Map<String, Vector<int[]>> dataSet) {
		System.arraycopy(dataSet.keySet().toArray(new String[dataSet.keySet().size()]), 0, classNames, 0, classNames.length);
		
		int classId = 0;
		for(String clazz : dataSet.keySet()) {
			final int currentDocumentCount = dataSet.get(clazz).size();
			updateTotals(classId, currentDocumentCount);
			
			calculateFeatureCountsForClass(dataSet.get(clazz), classId);
			classId++;
		}
		
		estimateAPriori();
		estimateFeatureProbabilities();
	}

	@Override
	public String classify(int[] histogram) {
		final double[] probs = new double[classNames.length];
		for(int i = 0; i < probs.length; i++) {
			probs[i] = aPriori[i];
		}
		
		for(int i = 0; i < histogram.length; i++) {
			for(int clazz = 0; clazz < classNames.length; clazz++) {
				final int currentFeature = histogram[i];
				if(currentFeature != 0) {
					final double prob = getEstimatedFeatureProbability(clazz, i);
					probs[clazz] *= prob*currentFeature;
				}
			}
		}
		
		int max = getMaximumPropbabilityClass(probs);
		
		return classNames[max];
	}
	
	private void updateTotals(int classId, final int currentDocumentCount) {
		documentCounts[classId] = currentDocumentCount;
		totalDocuments += currentDocumentCount;
	}

	private void calculateFeatureCountsForClass(Vector<int[]> dataSet, int classId) {
		for(int[] currentFeatures : dataSet) {
			for(int currentFeature = 0; currentFeature < currentFeatures.length; currentFeature++) {
				final int currentFeatureCount = currentFeatures[currentFeature];
				featureCounts[classId][currentFeature] += currentFeatureCount;
				totalFeatureCounts[classId] += currentFeatureCount;
			}
		}
	}

	private void estimateAPriori() {
		for(int currentClass = 0; currentClass < classCnt; currentClass++) {
			aPriori[currentClass] = (float) documentCounts[currentClass] / totalDocuments;
		}
	}
	
	private void estimateFeatureProbabilities() {
		for(int i = 0; i < classCnt; i++) {
			for(int j = 0; j < featureCnt; j++) {
				probabilities[i][j] = estimateFeatureProbability(i, j);
			}
		}
	}
	
	/*
	 * Calculate P(F = feature | C = clazz) using add one smoothing.
	 */
	private double estimateFeatureProbability(int clazz, int feature) {
		return (double)(featureCounts[clazz][feature]+1) / totalFeatureCounts[clazz];
	}

	private double getEstimatedFeatureProbability(int clazz, int feature) {
		return probabilities[clazz][feature];
	}
	
	private int getMaximumPropbabilityClass(final double[] probs) {
		int max = 0;
		double maxValue = probs[max];
		
		for(int i = 1; i < probs.length; i++) {
			if(probs[i] > maxValue) {
				max = i;
				maxValue = probs[i];
			}
		}
		return max;
	}
	
}
