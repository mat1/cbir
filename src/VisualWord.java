import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

import mpi.cbg.fly.Feature;


public class VisualWord {
	
	//the Cebir FeatureVector
	public Feature centroied;
	
	public Set<Feature> points = Collections.synchronizedSet(new HashSet<Feature>());
	
	//the unique class ID
	public int	classID;
	
	//a placeholder for a class verification value
	public Object verificationValue;

	public VisualWord(Feature centroied, int classID) {
		this.centroied = centroied;
		this.classID = classID;
	}
	
}
