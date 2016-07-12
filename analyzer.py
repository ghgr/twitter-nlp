#from bson.objectid import ObjectId
#from pymongo import MongoClient
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot(data,xlabel,ylabel):
	plt.scatter(data[xlabel],data[ylabel],c=data.returns*100, s=100)
	plt.colorbar()
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

if __name__=="__main__":

#	client = MongoClient('mongodb://localhost:27017/')
#	db = client.sentiment
#	data = pd.DataFrame(list(db.tesla.find()))

	data = pd.read_csv("cache/results.csv")

	print "There are",data.shape[0],"elems"

	data['returns'] = data.AdaBoost_50-data.buy_and_hold_50
	data.sort_values("returns",inplace=True, ascending=False)

	authors = set(data.keys())-set(['_id','returns','buy_and_hold_50','AdaBoost_50'])
	for author in authors:
		print "%20s\t" % (author),
	print "%20s" % ("diff_pct")
	for elem in data.head(10).iterrows():
		elem = elem[1]
		for author in authors:
			print "%19.2f%%\t" % (elem[author]*100),
		print "%19.2f%%" % (elem['returns']*100)


	plt.subplot(221)
	plot(data,"elonmusk", "katyperry")
	plt.subplot(222)
	plot(data,"elonmusk", "teslamotors")
	plt.subplot(223)
	plot(data,"teslamotors", "katyperry")
	plt.subplot(224)
	plot(data,"teslamotors", "elonmusk")

	plt.show()
