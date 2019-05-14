import java.util.List;
import java.io.FileNotFoundException;
import java.util.Arrays;



public class kNNMain{

  public static void main(String[] args) throws FileNotFoundException{

    // TASK 1: Use command line arguments to point DataSet.readDataSet method to
    // the desired file. Choose a given DataPoint, and print its features and label
	//String path = args[0];
	//DataSet ds = new DataSet();
	double fractionTestSet = 0.6;
	double fractionTrainingTest = 1.0 - fractionTestSet;
	
	
	List<DataPoint> allMyData = DataSet.readDataSet("C:\\Users\\orian\\Desktop\\w19-360420-machine-learning\\data\\breastCancer.csv");
	DataPoint dp4 = allMyData.get(3);
	String dp4Label = dp4.getLabel();
	double[] dp4x = dp4.getX();
	System.out.println(dp4Label);
	System.out.println(Arrays.toString(dp4x));
	
	// System.out.println(path);


    //TASK 2:Use the DataSet class to split the fullDataSet into Training and Held Out Test Dataset
	List<DataPoint> testSet = DataSet.getTestSet(allMyData, fractionTestSet);
	List<DataPoint> trainingSet = DataSet.getTrainingSet(allMyData, fractionTrainingTest);

    // TASK 3: Use the DataSet class methods to plot the 2D data (binary and multi-class)
	//Don't do


    // TASK 4: write a new method in DataSet.java which takes as arguments to DataPoint objects,
    // and returns the Euclidean distance between those two points (as a double)
  /*	public static double (DataPoint dp1, DataPoint dp2)
	{
		double[] dp1x = dp1.getX();
		double[] dp2x = dp2.getX();
		double sum = 0;
		for (int i = 0; i < dp1x.length; i++)
		{
			sum += Math.pow(dp1x[i] - dp2x[i], 2);
		}
		return Math.sqrt(sum);
	}*/
	
    // TASK 5: Use the KNNClassifier class to determine the k nearest neighbors to a given DataPoint,
    // and make a print a predicted target label
	KNNClassifier kc = new KNNClassifier(7);
	DataPoint[] neighbours = kc.getNearestNeighbors(trainingSet, dp4);
	String predict = kc.predict(trainingSet, dp4);
	System.out.println(predict);
	System.out.println(Arrays.toString(neighbours));


    // TASK 6: loop over the datapoints in the held out test set, and make predictions for Each
    // point based on nearest neighbors in training set. Calculate accuracy of model.
	for(int i = 0; i < testSet.size(); i++)
	{
		String prediction = kc.predict(trainingSet, testSet.get(i));
		System.out.println(prediction);
	}

  }
  

}
