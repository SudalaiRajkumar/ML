import sys
from math import sqrt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import KFold
from sklearn import ensemble
from sklearn import linear_model as lm
from sklearn.metrics import mean_squared_error as mse
import xgboost as xgb

def runXGB(train_X, train_y, test_X, test_y=None):
        params = {}
        params["objective"] = "reg:linear"
        params["eta"] = 0.02
        params["min_child_weight"] = 8
        params["subsample"] = 0.9
        params["colsample_bytree"] = 0.8
        params["silent"] = 1
        params["max_depth"] = 8
        params["seed"] = 1
        plst = list(params.items())
        num_rounds = 500

        xgtrain = xgb.DMatrix(train_X, label=train_y)
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)
        pred_test_y = model.predict(xgtest)
        return pred_test_y

def rmse(act_y, pred_y):
	return np.sqrt(mse(act_y, pred_y))


if __name__ == "__main__":
	# Input data path #
        data_path = "../Data/"
        train_file = data_path + "Train_JPXjxg6.csv"
        test_file = data_path + "Test_mvj827l.csv"

	# Reading the csv file into pandas dataframe #
	train_df = pd.read_csv(train_file)
	test_df = pd.read_csv(test_file)

        print "Converting to date format"
        train_df["Date"] = (pd.to_datetime(train_df["Datetime"], format="%d-%m-%Y %H:%M"))
        test_df["Date"] = (pd.to_datetime(test_df["Datetime"], format="%d-%m-%Y %H:%M"))

	# Getting the dv and id values #
	train_y = np.array(train_df.Count.values)
	test_id = test_df.Datetime.values

        print "Processing Date field.."
        train_df["DayOfMonth"] = train_df["Date"].apply(lambda x: x.day)
        test_df["DayOfMonth"] = test_df["Date"].apply(lambda x: x.day)
	train_df["Hour"] = train_df["Date"].apply(lambda x: x.hour)
        test_df["Hour"] = test_df["Date"].apply(lambda x: x.hour)
	train_df["WeekDay"] = train_df["Date"].apply(lambda x: x.weekday())
	test_df["WeekDay"] = test_df["Date"].apply(lambda x: x.weekday())
	train_df["DayCount"] = train_df["Date"].apply(lambda x: x.toordinal())
        test_df["DayCount"] = test_df["Date"].apply(lambda x: x.toordinal())

	# Dropping the columns that are not needed #	
	train_df.drop(["Datetime","Date","Count"], axis=1, inplace=True)
	test_df.drop(["Datetime","Date"], axis=1, inplace=True)

	# Running the xgb model #
	preds = runXGB(np.array(train_df), train_y, np.array(test_df))
        preds = preds.astype('int')

	# Saving the predictions #
        sample = pd.read_csv(data_path + "Test_mvj827l.csv")
        sample["Count"] = preds
        sample.to_csv("sub_xgb.csv", index=False)
