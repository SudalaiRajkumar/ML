import sys
import operator
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, metrics, ensemble
import xgboost as xgb

def getCountVar(compute_df, count_df, var_name, count_var="v1"):
    grouped_df = count_df.groupby(var_name, as_index=False).agg('size').reset_index()
    grouped_df.columns = [var_name, "var_count"]
    merged_df = pd.merge(compute_df, grouped_df, how="left", on=var_name)
    merged_df.fillna(-1, inplace=True)
    return list(merged_df["var_count"])

def create_feature_map(features):
	outfile = open('xgb.fmap', 'w')
	for i, feat in enumerate(features):
		outfile.write('{0}\t{1}\tq\n'.format(i,feat))
	outfile.close()

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, extra_X=None, seed_val=0, num_rounds=200):
	params = {}
	params["objective"] = "binary:logistic"
	params['eval_metric'] = 'auc'
	params["eta"] = 0.02 
	params["subsample"] = 0.8
	params["min_child_weight"] = 5
	params["colsample_bytree"] = 0.7
	params["max_depth"] = 6
	params["silent"] = 1
	params["seed"] = seed_val

	plst = list(params.items())
	xgtrain = xgb.DMatrix(train_X, label=train_y)

	if test_y is not None:
		xgtest = xgb.DMatrix(test_X, label=test_y)
		watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
		model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=300)
	else:
		xgtest = xgb.DMatrix(test_X)
		model = xgb.train(plst, xgtrain, num_rounds)

	if feature_names is not None:
		create_feature_map(feature_names)
		model.dump_model('xgbmodel.txt', 'xgb.fmap', with_stats=True)
		importance = model.get_fscore(fmap='xgb.fmap')
		importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
		imp_df = pd.DataFrame(importance, columns=['feature','fscore'])
		imp_df['fscore'] = imp_df['fscore'] / imp_df['fscore'].sum()
		imp_df.to_csv("imp_feat.txt", index=False)

	pred_test_y = model.predict(xgtest)
	loss = 0

	if extra_X is not None:
		xgtest = xgb.DMatrix(extra_X)
		pred_extra_y = model.predict(xgtest)
		return pred_test_y, pred_extra_y, loss 

	if test_y is not None:
		loss = metrics.roc_auc_score(test_y, pred_test_y)
		print loss
		return pred_test_y, loss
	else:
	    return pred_test_y,loss

if __name__ == "__main__":
	## Reading the files and converting the dates ##
	data_path = "../input/Train/"
	train = pd.read_csv(data_path + "train_with_outcome.csv") 
	test = pd.read_csv(data_path + "Test.csv") 
	train['Registration_Date'].fillna('10-jan-90', inplace=True)
	test['Registration_Date'].fillna('10-jan-90', inplace=True)
	train['Registration_Date'] = pd.to_datetime(train['Registration_Date'], format="%d-%b-%y")
	test['Registration_Date'] = pd.to_datetime(test['Registration_Date'], format="%d-%b-%y")
	train['Registration_Date'] = train['Registration_Date'].apply(lambda x: x.toordinal())
	test['Registration_Date'] = test['Registration_Date'].apply(lambda x: x.toordinal())
	print train.shape, test.shape

	## Getting patient details and merging with train and test ##
	patient = pd.read_csv(data_path + "Patient_Profile.csv", na_values=['None',''])
	patient['First_Interaction'] = pd.to_datetime(patient['First_Interaction'], format="%d-%b-%y")
	patient['First_Interaction'] = patient['First_Interaction'].apply(lambda x: x.toordinal())
	print patient.shape
	train = train.merge(patient, on=['Patient_ID'], how='left')
	test = test.merge(patient, on=['Patient_ID'], how='left')
	print train.shape, test.shape
	
	## Getting health camp details and merging with train and test ##
	hc_details = pd.read_csv(data_path + "Health_Camp_Detail.csv")
	hc_ids = list(hc_details.Health_Camp_ID.values)
	hc_details['Camp_Start_Date'] = pd.to_datetime(hc_details['Camp_Start_Date'], format="%d-%b-%y")
	hc_details['Camp_End_Date'] = pd.to_datetime(hc_details['Camp_End_Date'], format="%d-%b-%y")
	hc_details['Camp_Start_Date'] = hc_details['Camp_Start_Date'].apply(lambda x: x.toordinal())
	hc_details['Camp_End_Date'] = hc_details['Camp_End_Date'].apply(lambda x: x.toordinal())
	hc_details['Camp_Duration_Days'] = hc_details['Camp_End_Date'] - hc_details['Camp_Start_Date']
	print hc_details.head()
	train = train.merge(hc_details, on=['Health_Camp_ID'], how='left')
	test = test.merge(hc_details, on=['Health_Camp_ID'], how='left')
	print train.shape, test.shape

	## Reading the camp files ##
	first_camp_details = pd.read_csv(data_path + "First_Health_Camp_Attended.csv")
	first_camp_details = first_camp_details[["Patient_ID","Health_Camp_ID","Donation","Health_Score"]]
	train = train.merge(first_camp_details, on=["Patient_ID","Health_Camp_ID"], how='left')
	third_camp_details = pd.read_csv(data_path + "Third_Health_Camp_Attended.csv")
	third_camp_details = third_camp_details[["Patient_ID","Health_Camp_ID","Number_of_stall_visited","Last_Stall_Visited_Number"]]	
	train = train.merge(third_camp_details, on=["Patient_ID","Health_Camp_ID"], how='left')
	train["Number_of_stall_visited"].fillna(0, inplace=True)
	train["Donation"].fillna(0, inplace=True)
	train["Health_Score"].fillna(0, inplace=True)
	print train.shape, test.shape


	## Filling NA with -99 ##
	train.fillna(-99, inplace=True)	
	test.fillna(-99, inplace=True)

	## print create additional features ##
	print "Getting additional features."
	train["Diff_CampStart_Registration"] = train["Camp_Start_Date"] - train["Registration_Date"]
	test["Diff_CampStart_Registration"] = test["Camp_Start_Date"] - test["Registration_Date"]

	train["Diff_CampEnd_Registration"] = train["Camp_End_Date"] - train["Registration_Date"]
	test["Diff_CampEnd_Registration"] = test["Camp_End_Date"] - test["Registration_Date"]

	train["Diff_Registration_FirstInteraction"] = train["Registration_Date"] - train["First_Interaction"]
	test["Diff_Registration_FirstInteraction"] = test["Registration_Date"] - test["First_Interaction"]

	train["Diff_CampStart_FirstInteraction"] = train["Camp_Start_Date"] - train["First_Interaction"]
	test["Diff_CampStart_FirstInteraction"] = test["Camp_Start_Date"] - test["First_Interaction"]
	print train.shape, test.shape

	## Getitng the cat columns and label encode them ##
	cat_columns = []
	for col in train.columns:
		if train[col].dtype == 'object':
			print col
			cat_columns.append(col)
			enc = preprocessing.LabelEncoder()
			full_list = list(train[col].values) + list(test[col].values)
			enc.fit(full_list)
			train[col] = enc.transform(list(train[col].values))
			test[col]  = enc.transform(list(test[col].values))

	# getting count #
	for col in ["Patient_ID", "Health_Camp_ID"]:
		print "Count : ", col
		full_df = pd.concat([train, test])
		train["Count_"+col] = getCountVar(train, full_df, col)
		test["Count_"+col] = getCountVar(test, full_df, col)


	## do sorting so as to compute the next variables ##
	train = train.sort_values(['Camp_Start_Date', 'Camp_End_Date', 'Patient_ID']).reset_index(drop=True)
	test = test.sort_values(['Camp_Start_Date', 'Camp_End_Date', 'Patient_ID']).reset_index(drop=True)
	print train.head()

	print "First pass to get necessary details.."
	people_camp_dict = {}
	people_date_dict = {}
	people_dv_dict = {}
	people_cat1_dict = {}
	people_cdate_dict = {}
	people_donation_dict = {}
	people_num_stall_dict = {}
	people_last_stall_dict = {}
	people_fscore_dict = {}
	for ind, row in train.iterrows():
		pid = row['Patient_ID']
		cid = row['Health_Camp_ID']
		reg_date = row['Registration_Date']
		dv = row['Outcome']
		cat1 = row['Category1']
		cdate = row['Camp_Start_Date']
		donation = row['Donation']
		num_stall = row['Number_of_stall_visited']
		fscore = row['Health_Score']
	
		tlist = people_camp_dict.get(pid,[])
		tlist.append(cid)
		people_camp_dict[pid] = tlist[:]

		tlist = people_date_dict.get(pid,[])
		tlist.append(reg_date)
		people_date_dict[pid] = tlist[:]

		tlist = people_dv_dict.get(pid, [])
		tlist.append(dv)
		people_dv_dict[pid] = tlist[:]

		tlist = people_donation_dict.get(pid, [])
		tlist.append(donation)
		people_donation_dict[pid] = tlist[:]
	
		tlist = people_num_stall_dict.get(pid, [])
		tlist.append(num_stall)
		people_num_stall_dict[pid] = tlist[:]

		tlist = people_fscore_dict.get(pid, [])
		tlist.append(fscore)
		people_fscore_dict[pid] = tlist[:]

		tlist = people_cat1_dict.get(pid, [])
		tlist.append(cat1)
		people_cat1_dict[pid] = tlist[:]

		tlist = people_cdate_dict.get(pid, [])
		tlist.append(cdate)
		people_cdate_dict[pid] = tlist[:]

	print "Creating features now using dict for train.."
	last_date_list = []
	last_dv_list = []
	last_cat1_list = []
	mean_dv_list = []
	last_cdate_list = []
	last_donation_list = []
	last_num_stall_list = []
	last_fscore_list=[]
	for ind, row in train.iterrows():
		pid = row['Patient_ID']
		reg_date = row['Registration_Date']
		cat1 = row['Category1']
		cid = row['Health_Camp_ID']
		cdate = row['Camp_Start_Date']

		camp_list = people_camp_dict[pid]
		for ind, camp in enumerate(camp_list):
			if camp == cid:
				use_index = ind
				break
	
		tlist = people_date_dict[pid][:use_index]
		if len(tlist)>0:
			last_date_list.append(reg_date-tlist[-1])
		else:
			last_date_list.append(-99)

		tlist = people_dv_dict[pid][:use_index]
		if len(tlist)>0:
			last_dv_list.append(tlist[-1])
			mean_dv_list.append(np.mean(tlist))
		else:
			last_dv_list.append(-99)
			mean_dv_list.append(-99)

		tlist = people_donation_dict[pid][:use_index]
		if len(tlist)>0:
			last_donation_list.append(np.sum(tlist))
		else:
			last_donation_list.append(-99)

		tlist = people_num_stall_dict[pid][:use_index]
		if len(tlist)>0:
			last_num_stall_list.append(np.sum(tlist))
		else:
			last_num_stall_list.append(-99)

		tlist = people_fscore_dict[pid][:use_index]
		if len(tlist)>0:
			last_fscore_list.append(np.mean([i for i in tlist if i!=0]))
		else:
			last_fscore_list.append(-99)

		tlist = people_cat1_dict[pid][:use_index]
		if len(tlist)>0:
			last_cat1_list.append(tlist[-1])
		else:
			last_cat1_list.append(-99)

		tlist = people_date_dict[pid][use_index+1:]
		if len(tlist)>0:
			last_cdate_list.append(reg_date-tlist[0]) 
		else:	
			last_cdate_list.append(-99)

	print last_fscore_list[:50]

	train["Last_Reg_Date"] = last_date_list[:]
	train["Mean_Outcome"] = mean_dv_list[:]
	train["Last_Cat1"] = last_cat1_list[:]
	train["Next_Reg_Date"] = last_cdate_list
	train["Sum_Donations"] = last_donation_list[:]
	train["Sum_NumStalls"] = last_num_stall_list[:]
	train["Mean_Fscore"] = last_fscore_list[:]
			
	print "Prepare dict using val.."
	for ind, row in test.iterrows():
		pid = row['Patient_ID']
		cid = row['Health_Camp_ID']
		reg_date = row['Registration_Date']
		cat1 = row['Category1']
		cdate = row['Camp_Start_Date']
		
		tlist = people_camp_dict.get(pid,[])
		tlist.append(cid)
		people_camp_dict[pid] = tlist[:]
		
		tlist = people_date_dict.get(pid,[])
		tlist.append(reg_date)
		people_date_dict[pid] = tlist[:]
		
		tlist = people_cat1_dict.get(pid, [])
		tlist.append(cat1)
		people_cat1_dict[pid] = tlist[:]

		tlist = people_cdate_dict.get(pid, [])
		tlist.append(cdate)
		people_cdate_dict[pid] = tlist[:]
	
	print "Creating features for val using dict.."	
	last_date_list = []
	last_dv_list = []
	last_cat1_list = []
	mean_dv_list = []
	last_cdate_list = []
	last_donation_list = []
	last_num_stall_list = []
	last_fscore_list = []
	for ind, row in test.iterrows():
		pid = row['Patient_ID']
		reg_date = row['Registration_Date']
		cat1 = row['Category1']
		cid = row['Health_Camp_ID']
		cdate = row['Camp_Start_Date']
		
		camp_list = people_camp_dict[pid]
		for ind, camp in enumerate(camp_list):
			if camp == cid:
				use_index = ind
				break
		
		tlist = people_date_dict[pid][:use_index]
		if len(tlist)>0:
			last_date_list.append(reg_date-tlist[-1])
		else:
			last_date_list.append(-99)
		
		tlist = people_dv_dict.get(pid, [])
		if len(tlist)>0:
			last_dv_list.append(tlist[-1])
			mean_dv_list.append(np.mean(tlist))
		else:
			last_dv_list.append(-99)
			mean_dv_list.append(-99)

		tlist = people_donation_dict.get(pid, [])
		if len(tlist)>0:
			last_donation_list.append(np.sum(tlist))
		else:
			last_donation_list.append(-99)

		tlist = people_num_stall_dict.get(pid, [])
		if len(tlist)>0:
			last_num_stall_list.append(np.sum(tlist))
		else:
			last_num_stall_list.append(-99)

		tlist = people_fscore_dict.get(pid, [])
		if len(tlist)>0:
			last_fscore_list.append(np.mean([i for i in tlist if i!=0]))
		else:
			last_fscore_list.append(-99)
		
		tlist = people_cat1_dict[pid][:use_index]
		if len(tlist)>0:
			last_cat1_list.append(tlist[-1])
		else:
			last_cat1_list.append(-99)

		tlist = people_date_dict[pid][use_index+1:]
		if len(tlist)>0:
			last_cdate_list.append(reg_date-tlist[0])
		else:
			last_cdate_list.append(-99)

	test["Last_Reg_Date"] = last_date_list[:]
	test["Mean_Outcome"] = mean_dv_list[:]
	test["Last_Cat1"] = last_cat1_list[:]
	test["Next_Reg_Date"] = last_cdate_list[:]
	test["Sum_Donations"] = last_donation_list[:]
	test["Sum_NumStalls"] = last_num_stall_list[:]
	test["Mean_Fscore"] = last_fscore_list[:]

	train.fillna(-99, inplace=True)
	test.fillna(-99, inplace=True)
	
	print "Getting dv and id values"
	train_y = train.Outcome.values

	## Columns to drop ##
	print "Dropping columns.."
	drop_cols = ["Camp_Start_Date", "Camp_End_Date", "Registration_Date"] #, "First_Interaction"]
	drop_cols = drop_cols + ["LinkedIn_Shared", "Facebook_Shared", "Twitter_Shared", "Online_Follower", "Var4"]
	train.drop(drop_cols, axis=1, inplace=True) 
	test.drop(drop_cols, axis=1, inplace=True) 
	print train.shape, test.shape

	# preparing train and test #
	print "Choose the columns to use.."
	xcols = [col for col in train.columns if col not in ["Outcome", "Health_Camp_ID", "Patient_ID", "Der_Var1", "Number_of_stall_visited","Last_Stall_Visited_Number", "Donation", "Health_Score", "Mean_Fscore"]]
	print xcols
	train_X = np.array(train[xcols])
	test_X = np.array(test[xcols])
	print train_X.shape, test_X.shape

	print "Final Model.."
	preds = 0
	for seed_val, num_rounds in [[0,200], [2016,250], [1323, 225]]:
		print seed_val, num_rounds
		temp_preds, loss = runXGB(train_X, train_y, test_X, feature_names=xcols, seed_val=seed_val, num_rounds=num_rounds)
		preds += temp_preds
	preds = preds/3.

	out_df = pd.DataFrame({"Patient_ID":test.Patient_ID.values})
	out_df["Health_Camp_ID"] = test.Health_Camp_ID.values
	out_df["Outcome"] =  preds
	out_df.to_csv("sub_srk.csv", index=False)
