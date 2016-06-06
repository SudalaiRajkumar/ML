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
        params["eta"] = 0.002
        params["min_child_weight"] = 1
        params["subsample"] = 0.9
        params["colsample_bytree"] = 0.8
        params["silent"] = 1
        params["max_depth"] = 8
        params["seed"] = 1
        plst = list(params.items())
        num_rounds = 900

        xgtrain = xgb.DMatrix(train_X, label=train_y)
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)
        pred_test_y = model.predict(xgtest)
        return pred_test_y

def rmse(act_y, pred_y):
	return np.sqrt(mse(act_y, pred_y))

if __name__ == "__main__":
        data_path = "../Data/"
        train_file = data_path + "Train_KQyJ5eh.csv"
        test_file = data_path + "Test_HmLwURQ.csv"

	train_df = pd.read_csv(train_file)
	test_df = pd.read_csv(test_file)

        print "Converting to date format"
        train_df["Date_mod"] = (pd.to_datetime(train_df["Date"], format="%d-%b-%y"))
        test_df["Date_mod"] = (pd.to_datetime(test_df["Date"], format="%d-%b-%y"))

	train_y = np.array(train_df.Number_SKU_Sold.values)
	train_y[train_y > 20000000] = 20000000
	test_id = test_df.Date.values

        print "Processing Dates.."
        train_df["DayOfMonth"] = train_df["Date_mod"].apply(lambda x: x.day)
        test_df["DayOfMonth"] = test_df["Date_mod"].apply(lambda x: x.day)
        train_df["Month"] = train_df["Date_mod"].apply(lambda x: x.month)
        test_df["Month"] = test_df["Date_mod"].apply(lambda x: x.month)
        #train_df["Year"] = train_df["Date"].apply(lambda x: x.year)
        #test_df["Year"] = test_df["Date"].apply(lambda x: x.year)
	#train_df["Hour"] = train_df["Date"].apply(lambda x: x.hour)
        #test_df["Hour"] = test_df["Date"].apply(lambda x: x.hour)
	train_df["WeekDay"] = train_df["Date_mod"].apply(lambda x: x.weekday())
	test_df["WeekDay"] = test_df["Date_mod"].apply(lambda x: x.weekday())
        #train_df["WeekNo"] = train_df["Date_mod"].apply(lambda x: x.isocalendar()[1])
        #test_df["WeekNo"] = test_df["Date_mod"].apply(lambda x: x.isocalendar()[1])
        train_df["DayOfYear"] = train_df["Date_mod"].apply(lambda x: x.timetuple().tm_yday)
        test_df["DayOfYear"] = test_df["Date_mod"].apply(lambda x: x.timetuple().tm_yday)
	train_df["DayCount"] = train_df["Date_mod"].apply(lambda x: x.toordinal())
        test_df["DayCount"] = test_df["Date_mod"].apply(lambda x: x.toordinal())
	
	

	train_df.drop(["Date_mod","Date","Number_SKU_Sold"], axis=1, inplace=True)
	test_df.drop(["Date_mod","Date"], axis=1, inplace=True)

	print train_df.shape, test_df.shape
	print train_df.head()
	print test_df.head()

	preds_xgb = runXGB(np.array(train_df)[299:,:], train_y[299:], np.array(test_df))

	
	reg = lm.LinearRegression()
	reg.fit(np.array(train_df)[:,:], train_y[:])
	preds_lm = reg.predict( np.array(test_df))

	train_y[train_y > 15000000] = 15000000
	preds = 0.8*preds_xgb + 0.2*preds_lm

	preds[357] = 70000000

	# Saving the predictions #
        sample = pd.read_csv(data_path + "Sample_Submission_6FjDs3p.csv")
        sample["Number_SKU_Sold"] = preds
        sample.to_csv("sub.csv", index=False)
