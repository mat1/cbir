import java.util.Map;
import java.util.Vector;

public class MyClassifier implements IClassifier {
	
	public MyClassifier() {
		
	}
	
	/* IMPLEMENT THIS METHOD 
	 * Classifies a VisualWordHistogram into an Image Class based on the learned model 
	 * @param histogram the given VisualWordHistogram
	 * @return the name of the Image-Class
	 */
	@Override
	public String classify(int[] histogram) {
		return "";
	}
	
	/* IMPLEMENT THIS METHOD 
	 *  * Learns a model based on the training data set 
	 * @param dataSet a list of VisualWordHistograms for each Image-Class
	 * store a model for the Classifier internal
	 */
	@Override
	public void learn(Map<String, Vector<int[]>> dataSet) {
		
	}

}
