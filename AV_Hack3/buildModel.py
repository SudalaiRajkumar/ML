# -*- coding: utf-8 -*-
"""
Code for Analytics Vidhya Online Hackathon 3.0 - Find the Next Brain Wong !
http://discuss.analyticsvidhya.com/t/online-hackathon-3-0-find-the-next-brain-wong/2838
__author__ : SRK
"""
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
sys.path.append("/home/sudalai/Softwares/xgboost-master/wrapper/")
import xgboost as xgb

if __name__ == "__main__":
	# setting the input path and reading the data into dataframe #
	data_path = "../Data/"
	train = pd.read_csv(data_path+"Train.csv")
	test = pd.read_csv(data_path+"Test.csv")

	## mapping the var8 with the  given data and create a new column ##
	var8_map_dict = {"HXYB":0, "HXYC":0, "HXYD":0, "HXYE":0, "HXYF":1, "HXFG":1, "HXYH":1, "HXYI":1, "HXYJ":2, "HXYK":2, "HXYL":2, "HXYM":3, "HXYN":3, "HXYO":3}
	train_var8_map = []
	for var_val in train["Var8"]:
		if var8_map_dict.has_key(var_val):
			train_var8_map.append(var8_map_dict[var_val])
		else:
			train_var8_map.append(4)  # just in case if the value is missing in dict, assign 4
	test_var8_map = []
	for var_val in test["Var8"]:
                if var8_map_dict.has_key(var_val):
                        test_var8_map.append(var8_map_dict[var_val])
                else:
                        test_var8_map.append(4)
	train["Var8Map"] = train_var8_map
	test["Var8Map"] = test_var8_map

	## categical column name list ##
	categorical_columns = ['Var4', 'institute_city', 'institute_state', 'Var8', 'institute_country', 'Var10', 'Var11', 'Var12', 'Var13', 'Var14', 'Var15', 'Instructor_Past_Performance', 'Instructor_Association_Industry_Expert', 'project_subject', 'subject_area', 'secondary_subject', 'secondary_area', 'Resource_Category', 'Resource_Sub_Category', 'Var23', 'Var24']

	## Getting the ID and DV from the data frame ##
	train_y = np.array(train["Project_Valuation"])
	train_y[train_y>6121] = 6121
	train_id = np.array(train["ID"])
	test_id = np.array(test["ID"])

	## Creating the IDVs from the train and test dataframe ##
	train_X = train.copy()
	test_X = test.copy()

	## Fill up the na values with -999 ##
	train_X = train_X.fillna(-999)
	test_X = test_X.fillna(-999)

	## One hot encoding the categorical variables ##
	for var in categorical_columns:
		lb = LabelEncoder()
		full_var_data = pd.concat((train_X[var],test_X[var]),axis=0).astype('str')
		lb.fit( full_var_data )
		train_X[var] = lb.transform(train_X[var].astype('str'))
		test_X[var] = lb.transform(test_X[var].astype('str'))

	## Dropping the unnecessary columns from IDVs ##
	train_X = np.array( train_X.drop(['ID','Project_Valuation'],axis=1) )
	test_X = np.array( test_X.drop(['ID','Unnamed: 26'],axis=1) )
	print "Train shape is : ",train_X.shape
	print "Test shape is : ",test_X.shape

	
	################################ MODEL BUILDING ##################################################
        print "Building RF1"
        reg = ensemble.RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_leaf=7, max_features="auto", n_jobs=4, random_state=0)
        reg.fit(train_X, train_y)
        pred_test_y_rf1 = reg.predict(test_X)

        print "Building RF2"
        reg = ensemble.RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_leaf=2, max_features=0.8, n_jobs=4, random_state=0)
        reg.fit(train_X, train_y)
        pred_test_y_rf2 = reg.predict(test_X)

        print "Building GB1"
        reg = ensemble.GradientBoostingRegressor(n_estimators=400, max_depth=7, min_samples_leaf=8, max_features=0.3, subsample=0.6, learning_rate=0.01, random_state=0)
        reg.fit(train_X, train_y)
        pred_test_y_gb1 = reg.predict(test_X)

        print "Building GB2"
        reg = ensemble.GradientBoostingRegressor(n_estimators=600, max_depth=6, min_samples_leaf=8, max_features=0.3, subsample=0.6, learning_rate=0.01, random_state=0)
        reg.fit(train_X, train_y)
        pred_test_y_gb2 = reg.predict(test_X)

	print "Building XGB1"
        params = {}
        params["objective"] = "reg:linear"
        params["eta"] = 0.005
        params["min_child_weight"] = 10
        params["subsample"] = 0.7
        params["colsample_bytree"] = 0.6
        params["scale_pos_weight"] = 0.8
        params["silent"] = 1
        params["max_depth"] = 5
        params["max_delta_step"]=2
        params["seed"] = 0
        plst = list(params.items())
        xgtrain = xgb.DMatrix(train_X, label=train_y)
        xgtest = xgb.DMatrix(test_X)
        num_rounds = 1100
        model = xgb.train(plst, xgtrain, num_rounds)
        pred_test_y_xgb1 = model.predict(xgtest)

	print "Building XGB2"
	params = {}
        params["objective"] = "reg:linear"
        params["eta"] = 0.005
        params["min_child_weight"] = 6
        params["subsample"] = 0.7
        params["colsample_bytree"] = 0.6
        params["scale_pos_weight"] = 0.8
        params["silent"] = 1
        params["max_depth"] = 8
        params["max_delta_step"]=2
        params["seed"] = 0
        plst = list(params.items())
        xgtrain = xgb.DMatrix(train_X, label=train_y)
        xgtest = xgb.DMatrix(test_X)
        num_rounds = 800
        model = xgb.train(plst, xgtrain, num_rounds)
        pred_test_y_xgb2 = model.predict(xgtest)

	## Averaging the six models ##
        pred_test_y = 0.15*pred_test_y_rf1 + 0.15*pred_test_y_rf2 + 0.15*pred_test_y_gb1 + 0.15*pred_test_y_gb2 + 0.2*pred_test_y_xgb1 + 0.2*pred_test_y_xgb2

	## Writing the submission file ##
        out_df = pd.DataFrame({"ID":test_id, "Project_Valuation":pred_test_y})
        out_df.to_csv("sub_2.csv", index=False)
		
