package edu.usc.mltrio.recommender;

import java.io.File;

import org.grouplens.lenskit.ItemScorer;
import org.grouplens.lenskit.baseline.BaselineScorer;
import org.grouplens.lenskit.baseline.ItemMeanRatingItemScorer;
import org.grouplens.lenskit.baseline.UserMeanBaseline;
import org.grouplens.lenskit.baseline.UserMeanItemScorer;
import org.grouplens.lenskit.core.LenskitConfiguration;
import org.grouplens.lenskit.core.LenskitRecommender;
import org.grouplens.lenskit.data.dao.EventDAO;
import org.grouplens.lenskit.data.text.TextEventDAO;
import org.grouplens.lenskit.knn.NeighborhoodSize;
import org.grouplens.lenskit.knn.user.NeighborFinder;
import org.grouplens.lenskit.knn.user.UserUserItemScorer;
import org.grouplens.lenskit.transform.normalize.DefaultUserVectorNormalizer;
import org.grouplens.lenskit.transform.normalize.MeanCenteringVectorNormalizer;
import org.grouplens.lenskit.transform.normalize.MeanVarianceNormalizer;
import org.grouplens.lenskit.transform.normalize.UserVectorNormalizer;
import org.grouplens.lenskit.transform.normalize.VectorNormalizer;

public class LensKitRecommender implements Recommend{

	LenskitRecommender rec ;
	
	public LensKitRecommender() throws Exception{
		LenskitConfiguration config = new LenskitConfiguration();
		// bind ItemScorer to UserUserItemScorer
		config.bind(ItemScorer.class).to(UserUserItemScorer.class);
		// let's use personalized mean rating as the baseline/fallback
		// predictor.
		// 2-step process:
		// First, use the user mean rating as the baseline scorer
		config.bind(BaselineScorer.class, ItemScorer.class).to(UserMeanItemScorer.class);

		// Second, use the item mean rating as the base for user means
		config.bind(UserMeanBaseline.class, ItemScorer.class).to(ItemMeanRatingItemScorer.class);

		// and normalize ratings by baseline prior to computing similarities
		config.within(UserVectorNormalizer.class).bind(VectorNormalizer.class).to(MeanCenteringVectorNormalizer.class);
		
		config.set(NeighborhoodSize.class).to(300);
		
		config.within(NeighborFinder.class).bind(UserVectorNormalizer.class).to(DefaultUserVectorNormalizer.class);
		config.within(NeighborFinder.class).bind(VectorNormalizer.class).to(MeanVarianceNormalizer.class);

		
		config.bind(EventDAO.class).to(TextEventDAO.ratings(new File(Constants.INVITED_INFO_TRAIN_MAHOUT_CSV),","));
		
		rec = LenskitRecommender.build(config);

	}

	public double predict(long userID, long quesID) throws Exception {

		return rec.getRatingPredictor().predict(userID, quesID);
	}
}
