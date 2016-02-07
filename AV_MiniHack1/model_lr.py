import sys
from math import sqrt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import KFold
from sklearn import ensemble, preprocessing
from sklearn import linear_model as lm
from sklearn.metrics import mean_squared_error as mse

def rmse(act_y, pred_y):
	return np.sqrt(mse(act_y, pred_y))

if __name__ == "__main__":
	# Data path of the input files #
        data_path = "../Data/"
        train_file = data_path + "Train_JPXjxg6.csv"
        test_file = data_path + "Test_mvj827l.csv"

	print "Reading the files into dataframes.."
	train_df = pd.read_csv(train_file)
	test_df = pd.read_csv(test_file)

        print "Converting to date format.."
        train_df["Date"] = (pd.to_datetime(train_df["Datetime"], format="%d-%m-%Y %H:%M"))
        test_df["Date"] = (pd.to_datetime(test_df["Datetime"], format="%d-%m-%Y %H:%M"))

	print "Getting the dv and id column.."
	train_y = np.array(train_df.Count.values)
	test_id = test_df.Datetime.values

        print "Creating variables from date field.."
        train_df["Year"] = train_df["Date"].apply(lambda x: x.year)
        test_df["Year"] = test_df["Date"].apply(lambda x: x.year)
	train_df["Hour"] = train_df["Date"].apply(lambda x: x.hour)
        test_df["Hour"] = test_df["Date"].apply(lambda x: x.hour)
	train_df["WeekDay"] = train_df["Date"].apply(lambda x: x.weekday())
	test_df["WeekDay"] = test_df["Date"].apply(lambda x: x.weekday())
	train_df["DayCount"] = train_df["Date"].apply(lambda x: x.toordinal())
        test_df["DayCount"] = test_df["Date"].apply(lambda x: x.toordinal())

	train = train_df.drop(["Datetime","Date","Count"], axis=1)
	test = test_df.drop(["Datetime","Date"], axis=1)

	print "One hot encoding.."
        temp_train_arr = np.empty([train.shape[0],0])
        temp_test_arr = np.empty([test.shape[0],0])
        cols_to_drop = []
        for var in train.columns:
                if var in ["Hour", "WeekDay"]:
                        print var
                        lb = preprocessing.LabelEncoder()
                        full_var_data = pd.concat((train[var],test[var]),axis=0).astype('str')
                        temp = lb.fit_transform(np.array(full_var_data))
                        train[var] = lb.transform(np.array( train[var] ).astype('str'))
                        test[var] = lb.transform(np.array( test[var] ).astype('str'))

                        cols_to_drop.append(var)
                        ohe = preprocessing.OneHotEncoder(sparse=False)
                        ohe.fit(temp.reshape(-1,1))
                        temp_arr = ohe.transform(train[var].reshape(-1,1))
                        temp_train_arr = np.hstack([temp_train_arr, temp_arr])
                        temp_arr = ohe.transform(test[var].reshape(-1,1))
                        temp_test_arr = np.hstack([temp_test_arr, temp_arr])

	train = train.drop(cols_to_drop, axis=1)
        test = test.drop(cols_to_drop, axis=1)
	train = np.hstack( [np.array(train),temp_train_arr]).astype("float")
        test = np.hstack( [np.array(test),temp_test_arr]).astype("float")
	print train.shape, test.shape

	# Use the lastest data #
	train_X = np.array(train)[16000:]
	train_y = train_y[16000:]
	test_X = np.array(test)

	# Train the linear model and predict on test data #
	reg = lm.LinearRegression()
	reg.fit(train_X, train_y)
	preds = reg.predict(test_X).astype('int')
	
	# writing to out file #
	sample = pd.read_csv(data_path + "Test_mvj827l.csv")
	sample["Count"] = preds
	sample.to_csv("sub_lr.csv", index=False)
