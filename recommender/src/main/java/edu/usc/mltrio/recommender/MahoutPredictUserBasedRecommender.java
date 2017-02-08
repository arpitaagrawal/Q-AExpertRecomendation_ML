package edu.usc.mltrio.recommender;

import java.io.File;

import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.Recommender;

/**
 * Run user based CF on test data
 */
public class MahoutPredictUserBasedRecommender implements Recommend{
	Recommender recommender ;
	
	public MahoutPredictUserBasedRecommender() throws Exception  {
		DataModel model = new FileDataModel(
				new File(Constants.INVITED_INFO_TRAIN_MAHOUT_CSV));

		recommender = new MahoutUserBasedRecommender(1024, 0.1).buildRecommender(model);

	}

	public double predict(long userID, long quesID) throws Exception{
		
		return recommender.estimatePreference(userID, quesID);
	}
}
