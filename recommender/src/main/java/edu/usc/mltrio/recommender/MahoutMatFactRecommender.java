package edu.usc.mltrio.recommender;

import java.io.File;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.recommender.svd.ALSWRFactorizer;
import org.apache.mahout.cf.taste.impl.recommender.svd.SVDRecommender;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.Recommender;

/**
 * Find best num_neigh and similarity matrix using validation
 * 
 * LOWER THE BETTER
 * 
 * numFeatures - 64 lambda - 1.0E-4 Acc - 0.3511877294688672
 * numFeatures - 64 lambda - 0.001 Acc - 0.19067660161192174
 * 
 * numFeatures - 16 lambda - 1.0E-5 Acc - 1.9206223071447528
numFeatures - 16 lambda - 1.0E-4 Acc - 0.4778584128588928
numFeatures - 16 lambda - 0.001 Acc - 0.23063901276558188
numFeatures - 16 lambda - 0.01 Acc - 0.15821677404183898
numFeatures - 32 lambda - 1.0E-5 Acc - 1.3967344276030973
numFeatures - 32 lambda - 1.0E-4 Acc - 0.3938753453566021
numFeatures - 32 lambda - 0.001 Acc - 0.22004050429783453
numFeatures - 32 lambda - 0.01 Acc - 0.1508279857389991
numFeatures - 64 lambda - 1.0E-5 Acc - 1.860971584323923
numFeatures - 64 lambda - 1.0E-4 Acc - 0.40138566167735484
numFeatures - 64 lambda - 0.001 Acc - 0.19399877313013533
numFeatures - 64 lambda - 0.01 Acc - 0.14611190250341966
numFeatures - 128 lambda - 1.0E-5 Acc - 1.1303028809706241
numFeatures - 128 lambda - 1.0E-4 Acc - 0.30393174377740173
numFeatures - 128 lambda - 0.001 Acc - 0.1826482290884484
numFeatures - 128 lambda - 0.01 Acc - 0.15133927458241486

numFeatures - 16 lambda - 0.01 iter - 20 Acc - 0.16061455300665378
numFeatures - 16 lambda - 0.01 iter - 30 Acc - 0.16164097173763944
numFeatures - 16 lambda - 0.01 iter - 100 Acc - 0.1615779936817114
numFeatures - 16 lambda - 0.1 iter - 20 Acc - 0.1197736989118323
numFeatures - 16 lambda - 0.1 iter - 30 Acc - 0.12076408633734449
numFeatures - 16 lambda - 0.1 iter - 100 Acc - 0.11867177328416294
numFeatures - 16 lambda - 1.0 iter - 20 Acc - 0.1182703567246463
numFeatures - 16 lambda - 1.0 iter - 30 Acc - 0.11839535952216841
numFeatures - 16 lambda - 1.0 iter - 100 Acc - 0.11893849374752434
numFeatures - 32 lambda - 0.01 iter - 20 Acc - 0.15415307254849842
numFeatures - 32 lambda - 0.01 iter - 30 Acc - 0.15499697915984542

numFeatures - 50 lambda - 0.1 iter - 20 Acc - 0.11948889487171543

numFeatures - 50 lambda - 0.1 iter - 5 Acc - 0.11915307252105443
numFeatures - 50 lambda - 1.0 iter - 5 Acc - 0.11903541617194832
numFeatures - 50 lambda - 1.5 iter - 5 Acc - 0.11991324696811731
numFeatures - 50 lambda - 0.1 iter - 10 Acc - 0.11873918970940481
numFeatures - 50 lambda - 1.0 iter - 10 Acc - 0.11926448821257787
numFeatures - 50 lambda - 1.5 iter - 10 Acc - 0.11828599050300052

numFeatures - 16 lambda - 2.0 iter - 10 Acc - 0.11854998583970487
numFeatures - 16 lambda - 2.0 iter - 20 Acc - 0.11769710720362969
numFeatures - 16 lambda - 5.0 iter - 10 Acc - 0.11660158493166403
numFeatures - 16 lambda - 5.0 iter - 20 Acc - 0.11982521351757693
numFeatures - 16 lambda - 10.0 iter - 10 Acc - 0.11625635574961983
numFeatures - 16 lambda - 10.0 iter - 20 Acc - 0.11695078439146012

 */
public class MahoutMatFactRecommender implements RecommenderBuilder{
	
	int numFeatures = 50; 
	double lambda = 0.1; 
	int numIterations = 50;
	
	public MahoutMatFactRecommender(int numFeatures, double lambda, int numIterations) {
		this.numFeatures = numFeatures;
		this.lambda = lambda;
		this.numIterations = numIterations;
	}

	public static void main(String[] args) throws Exception{
		DataModel model = new FileDataModel(new File(Constants.INVITED_INFO_TRAIN_MAHOUT_CSV));
		
		RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
		
		int [] numFeaturesArr = {16};
		double [] lambdaArr = {16, 22};
		int[] iterArr = {10,20};
				
		for(int numFeaturesI : numFeaturesArr){
			for(double lambdaI : lambdaArr ){
				for (int iterI : iterArr ){
					RecommenderBuilder builder = new MahoutMatFactRecommender(numFeaturesI, lambdaI, iterI);
					
					double result = evaluator.evaluate(builder, null, model, 0.8, 0.8);
					System.out.println("numFeatures - "+ numFeaturesI + " lambda - "+ lambdaI + " iter - "+ iterI + " Acc - " + result);
				}
			}
		}
	}

	public Recommender buildRecommender(DataModel model) throws TasteException {

		ALSWRFactorizer factorizer = new ALSWRFactorizer(model, numFeatures, lambda, numIterations);

		return new SVDRecommender(model, factorizer);
	}
}
