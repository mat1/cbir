import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.Vector;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_print_interface;
import libsvm.svm_problem;

public class SVMClassifier implements IClassifier {

	private static final boolean ESTIMATE_PARAMETERS = true;
	private static final boolean PRINT_SVM_INFO = false;
	
	private final int featureCount;
	private String[] classNames;
	
	private svm_model model;
	private double C;
	private double gamma;
	private int maxFreq = 0;
	
	public SVMClassifier(int features) {
		this.featureCount = features;
		svm.svm_set_print_string_function(new svm_print_interface() {
			@Override
			public void print(String toPrint) {
				if(PRINT_SVM_INFO) {
					System.out.println(toPrint);
				}
			}
		});
	}
	
	@Override
	public String classify(int[] histogram) {
		svm_node[] node = normalizeNode(toSVMNode(histogram));
		final double result = svm.svm_predict(model, node);
		
		if(result > 0.5) {
			return classNames[1];
		}else{
			return classNames[0];
		}
	}

	@Override
	public void learn(Map<String, Vector<int[]>> dataSet) {
		if(dataSet.size() != 2) throw new IllegalArgumentException("This SVM implementation only supports binary classification");
		
		classNames = new String[dataSet.size()];
		
		List<Vector<int[]>> items = new ArrayList<>();
		int itemCount = fillFeatureVecotrsToItemList(dataSet.entrySet(), items);
		
		double[] labels = new double[itemCount];
		svm_node[][] vectors = new svm_node[itemCount][featureCount];

		convertInputToLibSVMInput(labels, vectors, items);
		normalizeVectorSpace(vectors);
		
		trainSVM(itemCount, labels, vectors);
	}

	private void trainSVM(int itemCount, double[] labels, svm_node[][] vectors) {
		svm_problem prob = createLibSVMProblem(itemCount, labels, vectors);
		svm_parameter param = createLibSVMParameters(prob);
		model = svm.svm_train(prob, param);
	}

	private svm_parameter createLibSVMParameters(svm_problem prob) {
		svm_parameter param = getDefaultParameters();
		
		if(ESTIMATE_PARAMETERS) {
			estimateParameters(prob, param);
			
			this.C = param.C;
			this.gamma = param.gamma;
		} else {
			param.gamma = this.gamma;
			param.C = this.C;
		}
		
		return param;
	}

	private int fillFeatureVecotrsToItemList(Set<Entry<String, Vector<int[]>>> entrySet, List<Vector<int[]>> items) {
		int classId = 0;
		int itemCount = 0;
		for(Entry<String,Vector<int[]>> entry : entrySet) {
			classNames[classId] = entry.getKey();
			items.add(entry.getValue());
			
			itemCount += entry.getValue().size();
			
			classId++;
		}
		return itemCount;
	}

	private svm_problem createLibSVMProblem(int itemCount, double[] labels,
			svm_node[][] vectors) {
		svm_problem prob = new svm_problem();
		prob.l = itemCount;
		prob.y = labels;
		prob.x = vectors;
		return prob;
	}
	
	private void convertInputToLibSVMInput(double[] labels, svm_node[][] vectors, List<Vector<int[]>> items) {
		int currentCount = 0;
		int classId = 0;
		for(Vector<int[]> itemsList : items) {
			fillWith(labels, classId, currentCount, itemsList.size());
			
			for(int[] item : itemsList) {
				vectors[currentCount] = toSVMNode(item);
				currentCount++;
			}
			
			classId++;
		}
	}
	
	private void normalizeVectorSpace(svm_node[][] data) {
		for(int i = 0; i < data.length; i++) {
			normalizeNode(data[i]);
		}
	}

	private svm_node[] normalizeNode(svm_node[] node) {
		for(int j = 0; j < featureCount; j++) {
			node[j].value = node[j].value / maxFreq;
		}
		return node;
	}
	
	private void fillWith(double[] labels, double value, int offset, int amount) {
		for(int i = offset; i < offset+amount && i < labels.length; i++) {
			labels[i] = value;
		}
	}

	private svm_node[] toSVMNode(int[] features) {
		svm_node[] node = new svm_node[features.length];
		for(int i = 0; i < featureCount; i++) {
			node[i] = new svm_node();
			node[i].index = i;
			node[i].value = features[i];
			
			if(features[i] > maxFreq) {
				maxFreq = features[i];
			}
		}
		return node;
	}
	
	private void estimateParameters(svm_problem prob, svm_parameter param) {
		SVMGridSearch searcher = new SVMGridSearch();
		double[] params = searcher.estimateParameters(prob, param);
		param.C = params[0];
		param.gamma = params[1];
		
//		System.out.println("C : " + param.C);
//		System.out.println("Gamma : " + param.gamma);
	}
	
	private svm_parameter getDefaultParameters() {
		svm_parameter param = new svm_parameter();

		param.svm_type = svm_parameter.C_SVC;
		param.kernel_type = svm_parameter.RBF;
		param.degree = 3;
		param.gamma = 0;
		param.coef0 = 0;
		param.nu = 0.5;
		param.cache_size = 100;
		param.C = 128;
		param.eps = 1e-3;
		param.p = 0.1;
		param.shrinking = 0;
		param.probability = 0;
		param.nr_weight = 0;
		param.weight_label = new int[0];
		param.weight = new double[0];
		
		return param;
	}
}
