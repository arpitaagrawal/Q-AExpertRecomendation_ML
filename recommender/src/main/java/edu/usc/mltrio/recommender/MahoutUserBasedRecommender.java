package edu.usc.mltrio.recommender;

import java.io.File;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.svd.ALSWRFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.SVDRecommender;
import org.apache.mahout.cf.taste.impl.similarity.UncenteredCosineSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

/**
 * Find best num_neigh and similarity matrix using validation
 * Neigh - 1110 Acc - 0.6704911610004202
 * Neigh - 110 Acc - 0.6675216277313736
 * Neigh - 710 Acc - 0.6682084557050264
 * 
 * Neigh - 64 min_similarity - 0.0 Acc - 0.6498210621159797
 * Neigh - 64 min_similarity - 0.01 Acc - 0.6641040001442645
 * Neigh - 64 min_similarity - 0.05 Acc - 0.6438712743672923
 * Neigh - 128 min_similarity - 0.0 Acc - 0.6615997598185176
 * Neigh - 128 min_similarity - 0.01 Acc - 0.6568353151610219
 * Neigh - 128 min_similarity - 0.05 Acc - 0.6634960395791719
 * Neigh - 512 min_similarity - 0.0 Acc - 0.6680383943763619
 * Neigh - 512 min_similarity - 0.01 Acc - 0.6574531099672709
 * Neigh - 512 min_similarity - 0.05 Acc - 0.6592442161957328
 * Neigh - 1024 min_similarity - 0.0 Acc - 0.6468725403075101
 * Neigh - 1024 min_similarity - 0.01 Acc - 0.6512808289717522
 * Neigh - 1024 min_similarity - 0.05 Acc - 0.6536606152690998
 * 
 * Similarity TanimotoCoefficientSimilarity
 */
public class MahoutUserBasedRecommender implements RecommenderBuilder{
	int num_neigh = 1000;
	double min_similarity = 0.1;
	int numFeatures = 50; 
	double lambda = 0.1; 
	int numIterations = 50;
	
	public MahoutUserBasedRecommender(int num_neigh, double min_similarity) {
		this.num_neigh = num_neigh;
		this.min_similarity = min_similarity;
	}
	
	public MahoutUserBasedRecommender(int numFeatures, double lambda, int numIterations) {
		this.numFeatures = numFeatures;
		this.lambda = lambda;
		this.numIterations = numIterations;
	}

	public static void main(String[] args) throws Exception{
		DataModel model = new FileDataModel(new File(Constants.INVITED_INFO_TRAIN_MAHOUT_CSV));
		
		RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
		
		int [] num_neigh_arr = {64,128,512,1024};
		double [] min_similarity_arr = {0.0, 0.01, 0.05};
		
		for(int num_neigh : num_neigh_arr){
			for(double min_similarity : min_similarity_arr ){
				RecommenderBuilder builder = new MahoutUserBasedRecommender(num_neigh, min_similarity);
				
				double result = evaluator.evaluate(builder, null, model, 0.8, 0.8);
				System.out.println("Neigh - "+ num_neigh + " min_similarity - "+ min_similarity + " Acc - " + result);
			}
		}
	}

	public Recommender buildRecommender(DataModel model) throws TasteException {

//		TanimotoCoefficientSimilarity UncenteredCosineSimilarity  LogLikelihoodSimilarity
		UserSimilarity similarity = new UncenteredCosineSimilarity(model);
		
		UserNeighborhood neighborhood = new NearestNUserNeighborhood(num_neigh,min_similarity, similarity, model);
//		UserNeighborhood neighborhood = new ThresholdUserNeighborhood(min_similarity, similarity, model);
		
		return new GenericUserBasedRecommender(model, neighborhood, similarity);
	
	}
}
