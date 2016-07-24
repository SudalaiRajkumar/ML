import sys
import csv
import operator
import pandas as pd
import numpy as np
from sklearn import preprocessing
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold

data_path = "../input/"
train_file_name = "Train_pjb2QcD.csv"
test_file_name = "Test_wyCirpO.csv"

def getCountVar(compute_df, count_df, var_name, count_var="Manager_Num_Application"):
        grouped_df = count_df.groupby(var_name, as_index=False)[count_var].agg('count')
        grouped_df.columns = [var_name, "var_count"]
        merged_df = pd.merge(compute_df, grouped_df, how="left", on=var_name)
        merged_df.fillna(-1, inplace=True)
        return list(merged_df["var_count"])

def create_feature_map(features):
        outfile = open('xgb.fmap', 'w')
        for i, feat in enumerate(features):
                outfile.write('{0}\t{1}\tq\n'.format(i,feat))
        outfile.close()

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0):
        params = {}
        params["objective"] = "binary:logistic"
        params['eval_metric'] = 'auc'
        params["eta"] = 0.01 #0.00334
        params["min_child_weight"] = 1
        params["subsample"] = 0.8
        params["colsample_bytree"] = 0.3
        params["silent"] = 1
        params["max_depth"] = 6
        params["seed"] = seed_val
        #params["max_delta_step"] = 2
        #params["gamma"] = 0.5
        num_rounds = 1000 #2500

        plst = list(params.items())
        xgtrain = xgb.DMatrix(train_X, label=train_y)

        if test_y is not None:
                xgtest = xgb.DMatrix(test_X, label=test_y)
                watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
                model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=500)
        else:
                xgtest = xgb.DMatrix(test_X)
                model = xgb.train(plst, xgtrain, num_rounds)

        if feature_names:
                        create_feature_map(feature_names)
                        model.dump_model('xgbmodel.txt', 'xgb.fmap', with_stats=True)
                        importance = model.get_fscore(fmap='xgb.fmap')
                        importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
                        imp_df = pd.DataFrame(importance, columns=['feature','fscore'])
                        imp_df['fscore'] = imp_df['fscore'] / imp_df['fscore'].sum()
                        imp_df.to_csv("imp_feat.txt", index=False)

        pred_test_y = model.predict(xgtest)

        if test_y is not None:
                loss = roc_auc_score(test_y, pred_test_y)
                print loss
        	return pred_test_y, loss
	else:
		return pred_test_y


if __name__ == "__main__":
	print "Reading files.."
        train = pd.read_csv(data_path + train_file_name)
        test = pd.read_csv(data_path + test_file_name)
        print train.shape, test.shape

	print "Rank vars.."
        prev_date = 0
        count_dict = {}
        for name, row in train.iterrows():
                count_dict[ row["Application_Receipt_Date"] ] = count_dict.get(row["Application_Receipt_Date"],0) + 1
        for name, row in test.iterrows():
                count_dict[ row["Application_Receipt_Date"] ] = count_dict.get(row["Application_Receipt_Date"],0) + 1

        prev_date = 0
        rank_list = []
        count_list = []
        rankpct_list = []
        for name, row in train.iterrows():
                date_value = row["Application_Receipt_Date"]
                if date_value != prev_date:
                        rank = 1
                        prev_date = date_value
                else:
                        rank += 1
                rank_list.append( rank )
                count_list.append( count_dict[date_value] )
                rankpct_list.append( float(rank) / count_dict[date_value] )
        train["dayrank"] = rank_list[:]
        train["daycount"] = count_list[:]
        train["dayrankpct"] = rankpct_list[:]

        prev_date = 0
        rank_list = []
        count_list = []
        rankpct_list = []
        for name, row in test.iterrows():
                date_value = row["Application_Receipt_Date"]
                if date_value != prev_date:
                        rank = 1
                        prev_date = date_value
                else:
                        rank += 1
                rank_list.append( rank )
                count_list.append( count_dict[date_value] )
                rankpct_list.append( float(rank) / count_dict[date_value] )
        test["dayrank"] = rank_list[:]
        test["daycount"] = count_list[:]
        test["dayrankpct"] = rankpct_list[:]
        print train.dayrank.describe()
        print test.dayrank.describe()

        print "Getting DV and ID.."
        train_y = train.Business_Sourced.values
        train_ID = train.ID.values
        test_ID = test.ID.values

	print "New feats.."
	print "Some more features.."
        new_feats = ["DOJ_DOB", "DOB_Applicant_Gender", "DOB_Qualification", "DOB_Gender_Qual"] 
        train["DOJ_DOB"] = train["Manager_DOJ"].astype('str') + "_" + train["Manager_DoB"].astype('str')
        train["DOB_Applicant_Gender"] = train["Manager_DoB"].astype('str') + "_" + train["Applicant_Gender"].astype('str')
        train["DOB_Qualification"] = train["Manager_DoB"].astype('str') + "_" + train["Applicant_Qualification"].astype('str')
	train["DOB_Gender_Qual"] = train["Manager_DoB"].astype('str') + "_" + train["Applicant_Gender"].astype('str') + "_" + train["Applicant_Qualification"].astype('str')
        test["DOJ_DOB"] = test["Manager_DOJ"].astype('str') + "_" + test["Manager_DoB"].astype('str')
        test["DOB_Applicant_Gender"] = test["Manager_DoB"].astype('str') + "_" + test["Applicant_Gender"].astype('str')
        test["DOB_Qualification"] = test["Manager_DoB"].astype('str') + "_" + test["Applicant_Qualification"].astype('str')
	test["DOB_Gender_Qual"] = test["Manager_DoB"].astype('str') + "_" + test["Applicant_Gender"].astype('str') + "_" + test["Applicant_Qualification"].astype('str')

	print "Label encoding.."
	cat_columns = ["Applicant_Gender", "Applicant_Marital_Status", "Applicant_Occupation", "Applicant_Qualification", "Manager_Joining_Designation", "Manager_Current_Designation", "Manager_Status", "Manager_Gender"]
	for f in cat_columns + new_feats:
                        print(f), len(np.unique(train[f].values))
                        lbl = preprocessing.LabelEncoder()
                        lbl.fit(list(train[f].values) + list(test[f].values))
                        train[f] = lbl.transform(list(train[f].values))
                        test[f] = lbl.transform(list(test[f].values))
                        new_train = pd.concat([ train[['Manager_Num_Application',f]], test[['Manager_Num_Application',f]] ])
                        train["CountVar_"+str(f)] = getCountVar(train[['Manager_Num_Application',f]], new_train[['Manager_Num_Application', f]], f)
                        test["CountVar_"+str(f)] = getCountVar(test[['Manager_Num_Application',f]], new_train[['Manager_Num_Application',f]], f)

	print "Working on dates.."
	for date_col in ["Application_Receipt_Date", "Applicant_BirthDate", "Manager_DOJ", "Manager_DoB"]:
		print date_col
		train[date_col].fillna("1/1/1900", inplace=True)
		test[date_col].fillna("1/1/1900", inplace=True)
		train[date_col] = (pd.to_datetime(train[date_col], format="%m/%d/%Y"))
        	test[date_col] = (pd.to_datetime(test[date_col], format="%m/%d/%Y"))
		train[date_col] = train[date_col].apply(lambda x: x.toordinal())
	        test[date_col] = test[date_col].apply(lambda x: x.toordinal())

	dev_index = np.where(train["Application_Receipt_Date"]<=733100)[0]
	val_index = np.where(train["Application_Receipt_Date"]>733100)[0]
	print "Dropping unwanted cols.."
	drop_cols = []
	train.drop(["ID", "Business_Sourced"]+drop_cols, axis=1, inplace=True)
	test.drop(["ID"] + drop_cols, axis=1, inplace=True)

	print "Fill NA.."
	train.fillna(-999, inplace=True)
	test.fillna(-999, inplace=True)

	print "New features.."
        train["Manager_Business2"] = train["Manager_Business"] - train["Manager_Business2"]
        test["Manager_Business2"] = test["Manager_Business"] - test["Manager_Business2"]
        train["Manager_Num_Products2"] = train["Manager_Num_Products"] - train["Manager_Num_Products2"]
        test["Manager_Num_Products2"] = test["Manager_Num_Products"] - test["Manager_Num_Products2"]

	print "Converting to array.."
	feat_names = list(train.columns)
	train = np.array(train)
	test = np.array(test)
	print train.shape, test.shape
	assert train.shape[1] == test.shape[1]

	full_preds = 0
	for rs in [1, 1343, 445234]:
	        preds = runXGB(train, train_y, test, feature_names=feat_names, seed_val = rs)
		full_preds += preds
	full_preds /= 3.
	
	out_df = pd.DataFrame({"ID":test_ID})
	out_df["Business_Sourced"] = full_preds
	out_df.to_csv("final.csv", index=False)
