# -*- coding: utf-8 -*-
"""
Benchmark script for Analytics Vidhya Online Hackathon using Linear Regression. 
__author__ : SRK
Date : July 11, 2015
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
	## specify the location of input files ##
	data_path = "../Data/"
	train_file = data_path + "train.csv"
	test_file = data_path + "test.csv"
	names_categorical = ['Category_article', 'Day_of_publishing']

	## creating pandas data frame for train and test ##
	train = pd.read_csv(train_file)
	test = pd.read_csv(test_file)

	# strpping the leading space in column names (some of them have leading spaces while reading using pandas read_csv) #
	train.columns =  [i.strip() for i in list(train.columns.values)]
	test.columns = [i.strip() for i in list(test.columns.values)]

	## getting the DV and ID values ##
	train_y = train["shares"]
	train_id = train["id"]
	test_id = test["id"]

	## dropping the categorical columns, ID and DV from dataframe ##
	train_X = train.drop( ["id"]+names_categorical+["shares"], axis=1)
	test_X = test.drop( ["id"]+names_categorical, axis=1)
	print "Train, test shape : ", train_X.shape, test_X.shape

	## building a linear regression model and predicting on test set ##
	lm_model = LinearRegression()
	lm_model.fit(train_X, train_y)
	pred_test_y = lm_model.predict(test_X)

	## Writing it to output csv files ##
	out_df = pd.DataFrame({"id":test_id, "predictions":pred_test_y})
	out_df.to_csv("benchmark.csv", index=False)
