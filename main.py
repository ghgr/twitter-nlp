import sys
import spacy
import operator
import numpy as np
import pandas as pd
import datetime
import random
from matplotlib import pyplot as plt
import dateutil
from sklearn.cross_validation import ShuffleSplit
from sklearn.ensemble import AdaBoostRegressor
import traceback
import cPickle
from bson.objectid import ObjectId
from pymongo import MongoClient



def getVectorFromTweet(tweet, cached_vectors):

	if tweet in cached_vectors:
		return cached_vectors[tweet]
	try:
		q = nlp(unicode(tweet, errors='ignore')).vector
	except:
		q = np.zeros(300)

	raise Exception("No tweet in cache and no lamguage model loaded")
	cached_vectors[tweet] = q

	with open("cache/tweets_vectos_dict.cPickle","w") as w:
		cPickle.dump(cached_vectors, w)	

	return q
		

def getTweets(authors):

	tweets = {}
	for author in authors:
		with open("data/tweets/"+author+".txt") as f:
			lines = [l.replace("\n","").split("|",2) for l in f.readlines()]
		for line in lines:
			_id, date, tweet = line
			date = dateutil.parser.parse(date).replace(hour = 0, minute=0, second = 0)
			if date not in tweets:
				tweets[date] = {}
			if author not in tweets[date]:
				tweets[date][author] =  []
			tweets[date][author].append(tweet)
	return tweets


def getAverageVectorForTweetsByDay(tweets_dict, weights_dict, cached_vectors):
	

	average_tweet = {}
	for date, tweets in tweets_dict.iteritems():
		vectors = np.zeros(300)
		present_authors_today = tweets.keys()
		total_weights_today = np.sum( weights_dict[p] for p in present_authors_today )
		if total_weights_today<0.01:
			continue
		for author in present_authors_today:
			vectors += weights_dict[author] * np.mean(np.array([getVectorFromTweet(tweet, cached_vectors) for tweet in tweets[author]]),axis=0)/total_weights_today

		average_tweet[date] = vectors
	return pd.DataFrame(average_tweet).transpose() 


def getStockReturns(company):
	data = pd.read_csv("data/stocks/"+company+".csv", sep=",", index_col=0)
	returns =  data.Close.diff()/data.Close.shift()
	returns.index = returns.index.astype(np.datetime64)
	returns.name = "returns"
	returns = returns.shift(-1) # We want the returns for NEXT day	
	returns += 1.0 # returns are from 0 to apply cumprod
	return returns


def geometricMean(v):
	if len(v)==0:
		return np.nan
	return np.prod(v)**(1.0/len(v))


def getMedianProfitRespectToBuyandHold(weights, cached_vectors, threshold_buy, company, tweets, returns, N_ITER=10, verbose=False):

	vectors = getAverageVectorForTweetsByDay(tweets, weights, cached_vectors) 
	returns.index.tz = vectors.index.tz
	data = pd.concat([vectors, returns], axis=1)
	data.dropna(inplace=True)


	X = data.drop('returns', axis=1)
	y = data.returns

	cv = ShuffleSplit(n = X.shape[0], n_iter=N_ITER, test_size = 0.5)
	
	clfs = [
			['AdaBoost', AdaBoostRegressor()]
		]


	results = {}
	if verbose:
		print "%20s\t%20s\t%20s\t%20s\t%20s" % ("Regressor", "Potential_Trades", "Potential_Buys", "Actual Buys", "Profit")
	for counter,(train_idx, test_idx) in enumerate(cv):
		if verbose:
			print "Iteration",counter+1,"of",N_ITER
		Xtrain = np.array(X.iloc[train_idx])
		Xtest = np.array(X.iloc[test_idx])
		ytrain = np.array(y.iloc[train_idx])
		ytest = np.array(y.iloc[test_idx])
		if verbose:
			print ""
		actions = np.zeros(ytest.shape)
		actions[ytest>1.0] = 1
		performance = geometricMean(ytest[actions>0])-1.0
		results.setdefault("perfect",[])
		results['perfect'].append(performance)
		if verbose:
			print "%20s\t%20d\t%20d\t%20d\t%19.2f%%" % ("Perfect", ytest.shape[0], (ytest>1.0).sum(), (actions>0).sum(), performance*100)

		actions = np.ones(ytest.shape)
		performance = geometricMean(ytest)-1.0
		results.setdefault("buy_and_hold",[])
		results['buy_and_hold'].append(performance)
		if verbose:
			print "%20s\t%20d\t%20d\t%20d\t%19.2f%%" % ("Buy_and_Hold", ytest.shape[0], (ytest>1.0).sum(), (actions>0).sum(), performance*100) 

		for name,clf in clfs:
			clf.fit(Xtrain, ytrain)
			ypred = clf.predict(Xtest)
			actions = np.zeros(ypred.shape)
			actions[ypred>threshold_buy] = 1
			performance = geometricMean(ytest[actions>0])-1.0
			results.setdefault(name,[])
			results[name].append(performance)
			if verbose:
				print "%20s\t%20d\t%20d\t%20d\t%19.2f%%" % (name, ytest.shape[0], (ytest>1.0).sum(), (actions>0).sum(), performance*100)

	diff = np.percentile(results['AdaBoost'],50)-np.percentile(results['buy_and_hold'],50)
	#print weights,": ",np.percentile(results['AdaBoost'],50),"-",np.percentile(results['buy_and_hold'],50),"=",diff

	writeToDB(weights, results)
			

def writeToDB(weights, results):
	data = weights.copy()
	data['buy_and_hold_50'] = np.percentile(results['buy_and_hold'],50)
	data['AdaBoost_50'] = np.percentile(results['AdaBoost'],50)
        client = MongoClient('mongodb://localhost:27017/')
        db = client.sentiment
	db.tesla.insert_one(data)


def getGrid(num_authors, elems):
	v = np.random.random((elems,num_authors))
	v/=v.sum(axis=1)[:,np.newaxis]
	return v

if __name__=="__main__":

	threshold_buy = 1.01
	N_ITER = 50
	N_RANDOM_WEIGHTS=100

	company = "NASDAQ_TSLA"
	weights = {'elonmusk' : 0.4, 'teslamotors' : 0.3, 'katyperry' : 0.3}
	

	print "Loading cache of vectors..."
	with open("cache/tweets_vectos_dict.cPickle") as f:
		cached_vectors = cPickle.load(f)	

	print "Done"

#	nlp = spacy.load('en')

	authors = weights.keys()

	tweets = getTweets(authors)
	returns = getStockReturns(company)


	grid = getGrid(len(authors), N_RANDOM_WEIGHTS)
	for weights_values in grid:

		weights = dict(zip(weights.keys(), weights_values))
		getMedianProfitRespectToBuyandHold(weights, cached_vectors, threshold_buy, company, tweets, returns, N_ITER, False)
