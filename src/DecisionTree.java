

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.Vector;



public class DecisionTree implements IClassifier {

	private static final double LOG2_BASE = Math.log(2);
	private static final double MIN_GAIN = 0.1;
	
	private final String[] classNames;
	private final int classCount;
	private final int featureCount;
	private DecisionTreeNode root;
	
	public DecisionTree(int classCnt, int featureCnt) {
		this.featureCount = featureCnt;
		this.classCount = classCnt;
		this.classNames = new String[classCnt];
	}
	
	@Override
	public String classify(int[] histogram) {
		int clazz = classify(histogram, root);
		
		return classNames[clazz];
	}

	@Override
	public void learn(Map<String, Vector<int[]>> dataSet) {
		List<LabeledHistogram> labeled = toLabeledList(dataSet);
		root = pruneTree(buildTree(labeled), MIN_GAIN);
	}

	private int classify(int[] histogram, DecisionTreeNode tree) {
		if(tree.isLeaf()) return tree.getClazz();
		
		int value = histogram[tree.getFeature()];
		DecisionTreeNode next;
		if(value < tree.getValue()) {
			next = tree.left;
		} else {
			next = tree.right;
		}
		
		return classify(histogram, next);
	}

 	private DecisionTreeNode pruneTree(DecisionTreeNode tree, double minGain) {
 		if(!tree.getLeft().isLeaf()) {
 			pruneTree(tree.getLeft(), minGain);
 		}
 		if(!tree.getRight().isLeaf()) {
 			pruneTree(tree.getRight(), minGain);
 		}
 		
 		if(tree.getLeft().isLeaf() && tree.getRight().isLeaf()) {
 			List<LabeledHistogram> combined = new ArrayList<>();
 			combined.addAll(tree.getLeft().getResults());
 			combined.addAll(tree.getRight().getResults());
 			
 			double delta = entropy(combined) - (entropy(tree.getLeft().getResults()) + entropy(tree.getRight().getResults()) / 2);
 			
 			if(delta < minGain) {
 				tree.toLeaf(combined);
 			}
 		}
 		
 		return tree;
 	}
	
	private DecisionTreeNode buildTree(List<LabeledHistogram> data) {
		if(data.size() == 0) return DecisionTreeNode.createLeafNode(null); 
		
		double currentScore = entropy(data);
		double bestGain = 0;
		int bestCriteria = 0;
		int bestValue = 0;
		Pair<List<LabeledHistogram>> bestSets = null;
		
		for(int col = 0; col < featureCount; col++) {
			Set<Integer> values = new HashSet<>();
			for(LabeledHistogram row : data) {
				values.add(row.histogram[col]);
			}
			
			for(int value : values) {
				Pair<List<LabeledHistogram>> sets = divideSet(data, col, value);
				
				double p = (double)sets.left.size() / data.size();
				double gain = currentScore - p*entropy(sets.left)-(1-p)*entropy(sets.right);
				if(gain > bestGain && sets.left.size() > 0 && sets.right.size() > 0) {
					bestGain = gain;
					bestCriteria = col;
					bestValue = value;
					bestSets = sets;
				}
			}
		}
		
		if(bestSets != null) {
			DecisionTreeNode left = buildTree(bestSets.left);
			DecisionTreeNode right = buildTree(bestSets.right);
			return DecisionTreeNode.createDecisionNode(left, right, bestCriteria, bestValue);
		}
		
		return DecisionTreeNode.createLeafNode(data);
	}
	
	private Pair<List<LabeledHistogram>> divideSet(List<LabeledHistogram> data, int col, int value) {
		List<LabeledHistogram> left = new ArrayList<>();
		List<LabeledHistogram> right = new ArrayList<>();
		
		for(LabeledHistogram row : data) {
			if(row.histogram[col] < value) {
				left.add(row);
			} else {
				right.add(row);
			}
		}
		
		return new Pair<>(left, right);
	}
	
	private List<LabeledHistogram> toLabeledList(Map<String, Vector<int[]>> dataSet) {
		List<LabeledHistogram> data = new ArrayList<>();
		
		int classId = 0;
		for(Entry<String,Vector<int[]>> entry : dataSet.entrySet()) {
			for(int[] hist : entry.getValue()) {
				data.add(new LabeledHistogram(hist, classId));
			}
			classNames[classId] = entry.getKey();
			classId++;
		}
		return data;
	}
	
	private int[] uniqueCounts(List<LabeledHistogram> classified) {
		int[] counts = new int[classCount];
		
		for(LabeledHistogram histo : classified) {
			counts[histo.label]++;
		}
		
		return counts;
	}
	
	private double entropy(List<LabeledHistogram> classified) {
		final int examplesCount = classified.size();
		final int[] counts = uniqueCounts(classified);
		
		double entropy = 0.0;
		for(int clazz = 0; clazz < counts.length; clazz++) {
			double probabilisticDensity = (double) counts[clazz] / examplesCount;
			entropy = entropy - probabilisticDensity * log2(probabilisticDensity);
		}
		
		return entropy;
	}

	private double log2(double value) {
		return Math.log(value)/LOG2_BASE;
	}
	
	private static class LabeledHistogram {
		public final int[] histogram;
		public final int label;
		
		public LabeledHistogram(int[] histogram, int label) {
			this.histogram = histogram;
			this.label = label;
		}
	}

	private static class Pair<T> {
		final T right;
		final T left;
		
		public Pair(T t, T v) {
			right = t;
			left = v;
		}
	}
	
	public static class DecisionTreeNode {
		private boolean isLeaf;
		private final int feature;
		private final int value;
		public List<LabeledHistogram> results;
		private DecisionTreeNode left;
		private DecisionTreeNode right;
		
		private DecisionTreeNode(List<LabeledHistogram> results, DecisionTreeNode left, DecisionTreeNode right, boolean isLeaf, int feature, int value) {
			this.isLeaf = isLeaf;
			this.feature = feature;
			this.value = value;
			this.left = left;
			this.right = right;
			this.results = results;
		}
		
		public boolean isLeaf() {
			return isLeaf;
		}

		public int getClazz() {
			int[] counts = new int[2];
			for(LabeledHistogram hist : results) {
				counts[hist.label]++;
			}
			int max = 0;
			int maxClass = 0;
			for(int i = 0; i < counts.length; i++) {
				if(max < counts[i]) {
					max = counts[i];
					maxClass = i;
				}
			}
			
			return maxClass;
		}

		public int getFeature() {
			return feature;
		}

		public int getValue() {
			return value;
		}

		public List<LabeledHistogram> getResults() {
			return results;
		}

		public DecisionTreeNode getLeft() {
			return left;
		}

		public DecisionTreeNode getRight() {
			return right;
		}
		
		public void toLeaf(List<LabeledHistogram> results) {
			isLeaf = true;
			left = null;
			right = null;
			this.results = results;
		}
		
		public static DecisionTreeNode createLeafNode(List<LabeledHistogram> data) {
			return new DecisionTreeNode(data, null, null, true, -1, -1);
		}
		
		public static DecisionTreeNode createDecisionNode(DecisionTreeNode left, DecisionTreeNode right, int feature, int value) {
			return new DecisionTreeNode(null, left, right, false, feature, value);
		}
	}
	
}
