import numpy as np
import pandas as pd
from sklearn import metrics, model_selection, ensemble, preprocessing, linear_model
import lightgbm as lgb

def getCountVar(compute_df, count_df, var_name, count_var="v1"):
	grouped_df = count_df.groupby(var_name)[count_var].agg('count').reset_index()
	grouped_df.columns = var_name + ["var_count"]

	merged_df = pd.merge(compute_df, grouped_df, how="left", on=var_name)
	merged_df.fillna(np.mean(grouped_df["var_count"].values), inplace=True)
	return list(merged_df["var_count"])

def getDVEncodeVar(compute_df, target_df, var_name, target_var="is_click", min_cutoff=1):
	if type(var_name) != type([]):
		var_name = [var_name]
	grouped_df = target_df.groupby(var_name)[target_var].agg(["mean"]).reset_index()
	grouped_df.columns = var_name + ["mean_value"]
	merged_df = pd.merge(compute_df, grouped_df, how="left", on=var_name)
	merged_df.fillna(np.mean(target_df[target_var].values), inplace=True)
	return list(merged_df["mean_value"])

def getDVEncodeVar2(compute_df, target_df, var_name, target_var="is_click", min_cutoff=1):
	if type(var_name) != type([]):
		var_name = [var_name]
	grouped_df = target_df.groupby(var_name)[target_var].agg(["sum"]).reset_index()
	grouped_df.columns = var_name + ["sum_value"]
	merged_df = pd.merge(compute_df, grouped_df, how="left", on=var_name)
	merged_df.fillna(np.mean(grouped_df["sum_value"].values), inplace=True)
	return list(merged_df["sum_value"])


def runLR(train_X, train_y, test_X, test_y=None, test_X2=None):
	model = linear_model.LogisticRegression(fit_intercept=True, C=0.3)
	model.fit(train_X, train_y)
	print model.coef_, model.intercept_
	train_preds = model.predict_proba(train_X)[:,1]
	test_preds = model.predict_proba(test_X)[:,1]
	test_preds2 = model.predict_proba(test_X2)[:,1]
	test_loss = 0
	if test_y is not None:
		train_loss = metrics.roc_auc_score(train_y, train_preds)
		test_loss = metrics.roc_auc_score(test_y, test_preds)
		print "Train and Test loss : ", train_loss, test_loss
	return test_preds, test_loss, test_preds2

def runET(train_X, train_y, test_X, test_y=None, test_X2=None, depth=10, leaf=5, feat=0.3):
	model = ensemble.ExtraTreesClassifier(
			n_estimators = 300,
					max_depth = depth,
					min_samples_split = 10,
					min_samples_leaf = leaf,
					max_features =  feat,
					n_jobs = 6,
					random_state = 0)
	model.fit(train_X, train_y)
	train_preds = model.predict_proba(train_X)[:,1]
	test_preds = model.predict_proba(test_X)[:,1]
	test_preds2 = model.predict_proba(test_X2)[:,1]
	test_loss = 0
	if test_y is not None:
		train_loss = metrics.roc_auc_score(train_y, train_preds)
		test_loss = metrics.roc_auc_score(test_y, test_preds)
		print "Depth, leaf, feat : ", depth, leaf, feat
		print "Train and Test loss : ", train_loss, test_loss
	return test_preds, test_loss, test_preds2

def runLGB(train_X, train_y, test_X, test_y=None, test_X2=None, feature_names=None, seed_val=0, rounds=500, dep=3, eta=0.001):
	params = {}
	params["objective"] = "binary"
	params['metric'] = 'auc'
	params["max_depth"] = dep
	params["min_data_in_leaf"] = 100
	params["learning_rate"] = eta
	params["bagging_fraction"] = 0.7
	params["feature_fraction"] = 0.7
	params["bagging_freq"] = 5
	params["bagging_seed"] = seed_val
	params["verbosity"] = -1
	num_rounds = rounds

	plst = list(params.items())
	lgtrain = lgb.Dataset(train_X, label=train_y)

	if test_y is not None:
		lgtest = lgb.Dataset(test_X, label=test_y)
		model = lgb.train(params, lgtrain, num_rounds, valid_sets=[lgtest], early_stopping_rounds=100, verbose_eval=20)
	else:
		lgtest = lgb.DMatrix(test_X)
		model = lgb.train(params, lgtrain, num_rounds)

	pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
	pred_test_y2 = model.predict(test_X2, num_iteration=model.best_iteration)

	loss = 0
	if test_y is not None:
		loss = metrics.roc_auc_score(test_y, pred_test_y)
		print loss
		return pred_test_y, loss, pred_test_y2
	else:
		return pred_test_y, loss, pred_test_y2

if __name__ == "__main__":
	print "Reading input files..."
	train_df = pd.read_csv("../input/train_feat.csv")
	test_df = pd.read_csv("../input/test_feat.csv")
	campaign_df = pd.read_csv("../input/campaign_data.csv")
	train_df["is_open_alone"] = train_df["is_click"].astype('float') / np.maximum(train_df["is_open"],1)
	print train_df.shape, test_df.shape
	print train_df.head()


	print np.sort(train_df["campaign_id"].unique())
	#camp_indices = [[range(29, 47), range(47,56)], [range(47,56), range(29, 47)]]

	print "Merging with campaign data.."
	train_df = pd.merge(train_df, campaign_df, on="campaign_id")
	test_df = pd.merge(test_df, campaign_df, on="campaign_id")
	print train_df.shape, test_df.shape
	kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

	train_y_open = train_df["is_open"].values
	train_y = train_df["is_click"].values
	test_id = test_df["id"].values
	train_unique_campaigns = np.array(train_df["campaign_id"].unique()) 
	cols_to_use = ["user_cum_count", "user_count", "user_date_diff", "user_camp_diff", "hour"] #, "total_links","no_of_internal_links","no_of_images","no_of_sections"]
	#cols_to_use = ["user_cum_count", "user_count", "user_camp_diff"]
	#cols_to_use = []
	#cols_to_use = cols_to_use + ["first_open", "first_click", "second_open", "second_click", "third_open", "third_click"]
	cols_to_use = cols_to_use + ["user_min_date", "user_mean_date", "user_max_date", "user_std_date"]
	cols_to_use = cols_to_use + ["camp_"+str(i) for i in range(29,81)] + ["camps_sent"]
	#cols_to_use = cols_to_use + ["user_std_date_click", "user_std_date_open"]
		
	#print "Label encoding.."
	#for c in ["communication_type"]:
	#		cols_to_use.append(c)
	#		lbl = preprocessing.LabelEncoder()
	#		lbl.fit(list(train_df[c].values.astype('str')) + list(test_df[c].values.astype('str')))
	#		train_df[c] = lbl.transform(list(train_df[c].values.astype('str')))
	#		test_df[c] = lbl.transform(list(test_df[c].values.astype('str')))
	
	
	#print "Full Count encoding.."
	#full_df = train_df.append(test_df)
	#print full_df.shape
	#for col in [["user_id"]]:
	#	if isinstance(col, list):
	#		col_name = "_".join(col)
	#	train_df[col_name + "_full_count"] = np.array( getCountVar(train_df, full_df, col, 'id'))
	#	test_df[col_name + "_full_count"] = np.array( getCountVar(test_df, full_df, col, 'id'))
	#	cols_to_use.append(col_name + "_full_count")

			
	print "Count encoding.."
	for col in [["user_id"], ["user_id", "communication_type"]]:
	#for col in [["user_id"]]:
		train_enc_values = np.zeros(train_df.shape[0])
		test_enc_values = 0
		for dev_index, val_index in kf.split(train_unique_campaigns):
		#for [dev_camp, val_camp] in camp_indices:
			dev_camp, val_camp = train_unique_campaigns[dev_index].tolist(), train_unique_campaigns[val_index].tolist()
			dev_X, val_X = train_df[train_df['campaign_id'].isin(dev_camp)], train_df[~train_df['campaign_id'].isin(dev_camp)]
			train_enc_values[train_df['campaign_id'].isin(val_camp)] = np.array( getCountVar(val_X[col], dev_X, col, 'is_click'))
			test_enc_values += np.array( getCountVar(test_df[col], dev_X, col, 'is_click'))
		test_enc_values /= 5.
		if isinstance(col, list):
			col = "_".join(col)
		train_df[col + "_count"] = train_enc_values
		test_df[col + "_count"] = test_enc_values
		cols_to_use.append(col + "_count")
		

		
	print "Target encoding.."
	for col in [["user_id"], ["user_id", "communication_type"]]:
	#for col in [["user_id"]]:
		train_enc_values = np.zeros(train_df.shape[0])
		test_enc_values = 0
		for dev_index, val_index in kf.split(train_unique_campaigns):
		#for [dev_camp, val_camp] in camp_indices:
			dev_camp, val_camp = train_unique_campaigns[dev_index].tolist(), train_unique_campaigns[val_index].tolist()
			dev_X, val_X = train_df[train_df['campaign_id'].isin(dev_camp)], train_df[~train_df['campaign_id'].isin(dev_camp)]
			train_enc_values[train_df['campaign_id'].isin(val_camp)] = np.array( getDVEncodeVar(val_X[col], dev_X, col, 'is_click'))
			test_enc_values += np.array( getDVEncodeVar(test_df[col], dev_X, col, 'is_click'))
		test_enc_values /= 5.
		if isinstance(col, list):
			col = "_".join(col)
		train_df[col + "_enc"] = train_enc_values
		test_df[col + "_enc"] = test_enc_values
		cols_to_use.append(col + "_enc")
	

	print "Open Target encoding.."
	for col in [["user_id"], ["user_id", "communication_type"]]:
	#for col in [["user_id"]]:
		train_enc_values = np.zeros(train_df.shape[0])
		test_enc_values = 0
		for dev_index, val_index in kf.split(train_unique_campaigns):
		#for [dev_camp, val_camp] in camp_indices:
			dev_camp, val_camp = train_unique_campaigns[dev_index].tolist(), train_unique_campaigns[val_index].tolist()
			dev_X, val_X = train_df[train_df['campaign_id'].isin(dev_camp)], train_df[~train_df['campaign_id'].isin(dev_camp)]
			train_enc_values[train_df['campaign_id'].isin(val_camp)] = np.array( getDVEncodeVar(val_X[col], dev_X, col, 'is_open'))
			test_enc_values += np.array( getDVEncodeVar(test_df[col], dev_X, col, 'is_open'))
		test_enc_values /= 5.
		if isinstance(col, list):
			col = "_".join(col)
		train_df[col + "_open_enc"] = train_enc_values
		test_df[col + "_open_enc"] = test_enc_values
		cols_to_use.append(col + "_open_enc")
			
	


	"""	
	print "Open Alone Target encoding.."
	#for col in [["user_id"], ["user_id", "communication_type"], ["user_id", "no_of_sections"]]:
	for col in [["user_id"]]:
		train_enc_values = np.zeros(train_df.shape[0])
		test_enc_values = 0
		for dev_index, val_index in kf.split(train_unique_campaigns):
			dev_camp, val_camp = train_unique_campaigns[dev_index].tolist(), train_unique_campaigns[val_index].tolist()
			dev_X, val_X = train_df[train_df['campaign_id'].isin(dev_camp)], train_df[~train_df['campaign_id'].isin(dev_camp)]
			train_enc_values[train_df['campaign_id'].isin(val_camp)] = np.array( getDVEncodeVar2(val_X[col], dev_X, col, 'is_open'))
			test_enc_values += np.array( getDVEncodeVar2(test_df[col], dev_X, col, 'is_open'))
		test_enc_values /= 5.
		if isinstance(col, list):
			col = "_".join(col)
		train_df[col + "_open_sum_enc"] = train_enc_values
		test_df[col + "_open_sum_enc"] = test_enc_values
		cols_to_use.append(col + "_open_sum_enc")	
	"""
	
	
	print cols_to_use
	train_X = train_df[cols_to_use]
	test_X = test_df[cols_to_use]
	print train_X.describe()
	print test_X.describe()

	#train_X.fillna(-1, inplace=True)
	#test_X.fillna(-1, inplace=True)	

	print "Model building.."
	model_name = "LGB"
	cv_scores = []
	pred_test_full = 0
	pred_val_full = np.zeros(train_df.shape[0])	
	for dev_index, val_index in kf.split(train_unique_campaigns):
	#for [dev_camp, val_camp] in camp_indices:
		dev_camp, val_camp = train_unique_campaigns[dev_index].tolist(), train_unique_campaigns[val_index].tolist()
		dev_X, val_X = train_X[train_df['campaign_id'].isin(dev_camp)], train_X[train_df['campaign_id'].isin(val_camp)]
		dev_y, val_y = train_y[train_df['campaign_id'].isin(dev_camp)], train_y[train_df['campaign_id'].isin(val_camp)]
		print dev_X.shape, val_X.shape

		if model_name == "LGB":
			pred_val1, loss1, pred_test1 = runLGB(dev_X, dev_y, val_X, val_y, test_X, rounds=5000, dep=4)
			pred_val2, loss2, pred_test2 = runLGB(dev_X, dev_y, val_X, val_y, test_X, rounds=5000, dep=4, seed_val=2018)
			pred_val3, loss3, pred_test3 = runLGB(dev_X, dev_y, val_X, val_y, test_X, rounds=5000, dep=4, seed_val=9876)
			pred_val = (pred_val1 + pred_val2 + pred_val3)/3. 
			pred_test = (pred_test1 + pred_test2 + pred_test3)/3.
			loss = (loss1 + loss2 + loss3)/3. 
		elif model_name == "ET":
			pred_val, loss, pred_test = runET(dev_X, dev_y, val_X, val_y, test_X, depth=20, leaf=20, feat=0.3)
		elif model_name == "LR":
			pred_val, loss, pred_test = runLR(dev_X, dev_y, val_X, val_y, test_X)

		pred_test_full += pred_test
		pred_val_full[train_df['campaign_id'].isin(val_camp)] = pred_val
		loss = metrics.roc_auc_score(train_y[train_df['campaign_id'].isin(val_camp)], pred_val)
		cv_scores.append(loss)
		print cv_scores
	pred_test_full /= 5.
	print np.mean(cv_scores), metrics.roc_auc_score(train_y, pred_val_full)

	sub_df = pd.DataFrame({"id":test_id})
	sub_df["is_click"] = pred_test_full
	sub_df.to_csv("srk_sub47.csv", index=False)


	
