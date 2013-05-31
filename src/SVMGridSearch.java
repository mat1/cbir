
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import libsvm.svm;
import libsvm.svm_parameter;
import libsvm.svm_problem;

/**
 * Implements an automated grid search for parameters of an RBF-Kernel.
 * 
 * This grid search is adapted from the svm-grid tool. It performes a brute-force
 * search over a fixed range of C and gamma values. The accuracy of the selected parameters
 * is measured using a n-fold cross validation.
 * 
 * To increase performance, only a subset of trainingvectors is used to perform the parameter
 * search. This leads to reasonable good results, but might be far away from the accuracy
 * of manual parameter search.
 * 
 * @author Florian Luescher
 * @author Matthias Brun
 */
public class SVMGridSearch {

	private static final int MAX_VECTORS = 2000;
	private static final int MAX_ITERATIONS = 200;
	private static final double ACCURACY_TRHESHOLD = 0.90;

	private static final int FOLD = 5;

	private static final int C_BEGIN = -5;
	private static final int C_END = 15;
	private static final int C_STEP = 2;

	private static final int G_BEGIN = -15;
	private static final int G_END = 3;
	private static final int G_STEP = 2;

	public double[] estimateParameters(svm_problem prob, svm_parameter param) {
		/* double[0] = C, double[1] = gamma */
		double[] result = new double[2];
		int oldLength = prob.l;

		List<Integer> cValues = generateValues(C_BEGIN, C_END, C_STEP);
		List<Integer> gValues = generateValues(G_BEGIN, G_END, G_STEP);
		
		if (prob.l > MAX_VECTORS) {
			prob.l = MAX_VECTORS;
		}

		double maxAccuracy = 0;
		int iterations = 0;
		for (int cValue : cValues) {
			for (int gValue : gValues) {
				iterations++;

				param.C = Math.pow(2, cValue);
				param.gamma = Math.pow(2, gValue);

				double acc = calcAccuracy(prob, param);
				if (acc > maxAccuracy) {
					maxAccuracy = acc;
					result[0] = param.C;
					result[1] = param.gamma;
				}

				if (maxAccuracy >= ACCURACY_TRHESHOLD || iterations >= MAX_ITERATIONS) {
					System.out.println("Max acc: " + maxAccuracy);
					System.out.println("Iterations: " + iterations);
					
					prob.l = oldLength;
					return result;
				}
			}
		}

		prob.l = oldLength;
		return result;
	}

	private List<Integer> generateValues(int start, int end, int step) {
		List<Integer> results = new ArrayList<Integer>();

		for (int i = start; i <= end; i += step) {
			results.add(i);
		}

		Collections.shuffle(results);

		return results;
	}

	private double calcAccuracy(svm_problem prob, svm_parameter param) {
		double[] target = new double[prob.l];
		svm.svm_cross_validation(prob, param, FOLD, target);

		int totalCorrect = 0;
		for (int i = 0; i < prob.l; i++) {
			if (target[i] == prob.y[i]) {
				++totalCorrect;
			}
		}

		return (double) totalCorrect / prob.l;
	}

}
