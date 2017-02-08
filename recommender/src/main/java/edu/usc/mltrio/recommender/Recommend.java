package edu.usc.mltrio.recommender;

public interface Recommend {

	public double predict(long userID, long questionID) throws Exception;
}
