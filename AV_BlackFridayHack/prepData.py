import sys
import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn import ensemble
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
sys.path.append("/home/sudalai/Softwares/XGB_pointfour/xgboost-master/wrapper/")
import xgboost as xgb

gender_dict = {'F':0, 'M':1}
age_dict = {'0-17':0, '18-25':1, '26-35':2, '36-45':3, '46-50':4, '51-55':5, '55+':6}
city_dict = {'A':0, 'B':1, 'C':2}
stay_dict = {'0':0, '1':1, '2':2, '3':3, '4+':4}

def getCountVar(compute_df, count_df, var_name):
	grouped_df = count_df.groupby(var_name)
	count_dict = {}
	for name, group in grouped_df:
		count_dict[name] = group.shape[0]

	count_list = []
	for index, row in compute_df.iterrows():
		name = row[var_name]
		count_list.append(count_dict.get(name, 0))
	return count_list

if __name__ == "__main__":
	data_path = "../Data/"
	train_file = data_path + "train.csv"
	test_file = data_path +  "test.csv"

	train_df = pd.read_csv(train_file)
	test_df = pd.read_csv(test_file)
	print train_df.shape, test_df.shape

	train_df["Gender"] = train_df["Gender"].apply(lambda x: gender_dict[x])
	test_df["Gender"] = test_df["Gender"].apply(lambda x: gender_dict[x])

	train_df["Age"] = train_df["Age"].apply(lambda x: age_dict[x])
	test_df["Age"] = test_df["Age"].apply(lambda x: age_dict[x])

	train_df["City_Category"] = train_df["City_Category"].apply(lambda x: city_dict[x])
        test_df["City_Category"] = test_df["City_Category"].apply(lambda x: city_dict[x])

	train_df["Stay_In_Current_City_Years"] = train_df["Stay_In_Current_City_Years"].apply(lambda x: stay_dict[x])
        test_df["Stay_In_Current_City_Years"] = test_df["Stay_In_Current_City_Years"].apply(lambda x: stay_dict[x])

	
	print "Getting count features.."
	train_df["Age_Count"] = getCountVar(train_df, train_df, "Age")
	test_df["Age_Count"] = getCountVar(test_df, train_df, "Age")
	print "Age", np.unique(test_df["Age_Count"])

	train_df["Occupation_Count"] = getCountVar(train_df, train_df, "Occupation")
        test_df["Occupation_Count"] = getCountVar(test_df, train_df, "Occupation")
        print "Occupation", np.unique(test_df["Occupation_Count"])

	train_df["Product_Category_1_Count"] = getCountVar(train_df, train_df, "Product_Category_1")
        test_df["Product_Category_1_Count"] = getCountVar(test_df, train_df, "Product_Category_1")
        print "Cat 1 ",np.unique(test_df["Product_Category_1_Count"])

	train_df["Product_Category_2_Count"] = getCountVar(train_df, train_df, "Product_Category_2")
        test_df["Product_Category_2_Count"] = getCountVar(test_df, train_df, "Product_Category_2")
        print "Cat 2 ", np.unique(test_df["Product_Category_2_Count"])

	train_df["Product_Category_3_Count"] = getCountVar(train_df, train_df, "Product_Category_3")
        test_df["Product_Category_3_Count"] = getCountVar(test_df, train_df, "Product_Category_3")
        print "Cat 3 ", np.unique(test_df["Product_Category_3_Count"])

	train_df["User_ID_Count"] = getCountVar(train_df, train_df, "User_ID")
        test_df["User_ID_Count"] = getCountVar(test_df, train_df, "User_ID")
        print "User id ", np.unique(test_df["User_ID_Count"])[:10]

	train_df["Product_ID_Count"] = getCountVar(train_df, train_df, "Product_ID")
        test_df["Product_ID_Count"] = getCountVar(test_df, train_df, "Product_ID")
        print "Product id ", np.unique(test_df["Product_ID_Count"])[:10]
	
	train_df.fillna(-999, inplace=True)
	test_df.fillna(-999, inplace=True)

	train_df.to_csv(data_path+"train_mod.csv", index=False)
	test_df.to_csv(data_path+"test_mod.csv", index=False)

