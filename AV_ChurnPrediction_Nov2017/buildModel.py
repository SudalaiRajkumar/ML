import sys
import random
import operator
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn import preprocessing, metrics, ensemble, neighbors, linear_model, tree, model_selection
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import manifold, decomposition
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

def create_feature_map(features):
	outfile = open('xgb.fmap', 'w')
	for i, feat in enumerate(features):
		outfile.write('{0}\t{1}\tq\n'.format(i,feat))
	outfile.close()

def runXGB(train_X, train_y, test_X, test_y=None, test_X2=None, feature_names=None, seed_val=0, rounds=500, dep=8, eta=0.05):
	params = {}
	params["objective"] = "binary:logistic"
	params['eval_metric'] = 'auc'
	params["eta"] = eta
	params["subsample"] = 0.7
	params["min_child_weight"] = 1
	params["colsample_bytree"] = 0.7
	params["max_depth"] = dep

	params["silent"] = 1
	params["seed"] = seed_val
	#params["max_delta_step"] = 2
	#params["gamma"] = 0.5
	num_rounds = rounds

	plst = list(params.items())
	xgtrain = xgb.DMatrix(train_X, label=train_y)

	if test_y is not None:
		xgtest = xgb.DMatrix(test_X, label=test_y)
		watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
		model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=100, verbose_eval=20)
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

	pred_test_y = model.predict(xgtest, ntree_limit=model.best_ntree_limit)
	pred_test_y2 = model.predict(xgb.DMatrix(test_X2), ntree_limit=model.best_ntree_limit)

	loss = 0
	if test_y is not None:
		loss = metrics.roc_auc_score(test_y, pred_test_y)
		return pred_test_y, loss, pred_test_y2
	else:
		return pred_test_y, loss, pred_test_y2

def runLGB(train_X, train_y, test_X, test_y=None, test_X2=None, feature_names=None, seed_val=0, rounds=500, dep=8, eta=0.05):
	params = {}
	params["objective"] = "binary"
	params['metric'] = 'auc'
	params["max_depth"] = dep
	params["min_data_in_leaf"] = 20
	params["learning_rate"] = eta
	params["bagging_fraction"] = 0.7
	params["feature_fraction"] = 0.7
	params["bagging_freq"] = 5
	params["bagging_seed"] = seed_val
	params["verbosity"] = 0
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

def runET(train_X, train_y, test_X, test_y=None, test_X2=None, depth=20, leaf=10, feat=0.2):
	model = ensemble.ExtraTreesClassifier(
			n_estimators = 100,
					max_depth = depth,
					min_samples_split = 2,
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

if __name__ == "__main__":
	#model_name = "ET"
	for model_name in ["LGB1", "XGB1"]:
		data_path = "../input/"
		train_df = pd.read_csv(data_path + "train.csv")
		test_df = pd.read_csv(data_path + "test.csv")

		# process columns, apply LabelEncoder to categorical features
		for c in train_df.columns:
			if train_df[c].dtype == 'object' and c not in ["Responders", "UCIC_ID"]:
				lbl = preprocessing.LabelEncoder()
				lbl.fit(list(train_df[c].values.astype('str')) + list(test_df[c].values.astype('str')))
				train_df[c] = lbl.transform(list(train_df[c].values.astype('str')))
				test_df[c] = lbl.transform(list(test_df[c].values.astype('str')))

		train_df.fillna(-99, inplace=True)
		test_df.fillna(-99, inplace=True)

		################### Feature Engineeering ###############################
		f1_f2_list = [["D_prev1", "D_prev2"], ["D_prev2", "D_prev3"], ["D_prev3", "D_prev4"], ["D_prev4", "D_prev5"], ["D_prev5", "D_prev6"],
					["CR_AMB_Prev1", "CR_AMB_Prev3"], ["CR_AMB_Prev1", "CR_AMB_Prev4"], ["CR_AMB_Prev1", "CR_AMB_Prev5"], ["CR_AMB_Prev1", "CR_AMB_Prev6"],
					["EOP_prev1", "CR_AMB_Prev1"], ["EOP_prev2", "CR_AMB_Prev2"], ["EOP_prev3", "CR_AMB_Prev3"], ["EOP_prev4", "CR_AMB_Prev4"], ["EOP_prev5", "CR_AMB_Prev5"], ["EOP_prev6", "CR_AMB_Prev6"],
					["EOP_prev1", "EOP_prev2"], ["EOP_prev2", "EOP_prev3"], ["EOP_prev3", "EOP_prev4"], ["EOP_prev4", "EOP_prev5"], ["EOP_prev5", "EOP_prev6"],
					["CR_AMB_Prev2", "CR_AMB_Prev4"], ["CR_AMB_Prev2", "CR_AMB_Prev5"], ["CR_AMB_Prev2", "CR_AMB_Prev6"],
					["EOP_prev1", "CR_AMB_Prev2"], ["EOP_prev1", "CR_AMB_Prev3"], ["EOP_prev1", "CR_AMB_Prev4"], ["EOP_prev1", "CR_AMB_Prev5"], ["EOP_prev1", "CR_AMB_Prev6"],
					["CR_AMB_Drop_Build_1", "CR_AMB_Drop_Build_2"], ["CR_AMB_Drop_Build_2", "CR_AMB_Drop_Build_3"], ["CR_AMB_Drop_Build_3", "CR_AMB_Drop_Build_4"],
					["BAL_prev1", "BAL_prev2"], ["BAL_prev2", "BAL_prev3"], ["BAL_prev3", "BAL_prev4"],
					["BAL_prev1", "CR_AMB_Prev1"], ["BAL_prev2", "CR_AMB_Prev2"], ["BAL_prev3", "CR_AMB_Prev3"],
					["I_AQB_PrevQ1", "I_AQB_PrevQ2"], ["I_NRV_PrevQ1", "I_NRV_PrevQ2"],
					["D_prev1", "D_prev3"], ["D_prev1", "D_prev4"], ["D_prev1", "D_prev6"],
					
					]
		for f1, f2 in f1_f2_list:
			train_df["Ratio_"+f1+"_"+f2] = train_df[f1].astype('float') / np.maximum(train_df[f2],1.)
			test_df["Ratio_"+f1+"_"+f2] = test_df[f1].astype('float') / np.maximum(test_df[f2],1.)


		print "Preparing response variable.."
		cols_to_leave = ["Responders", "UCIC_ID"]
		cols_to_use = [col for col in train_df.columns if col not in cols_to_leave]
		train_X = train_df[cols_to_use]
		test_X = test_df[cols_to_use]
		train_y = (train_df["Responders"]).values 
		train_id = train_df["UCIC_ID"].values
		test_id = test_df["UCIC_ID"].values

		print "Model building.."
		kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2018)
		cv_scores = []
		pred_test_full = 0
		pred_val_full = np.zeros(train_X.shape[0])
		for dev_index, val_index in kf.split(train_X):
			dev_X, val_X = train_X.iloc[dev_index,:], train_X.iloc[val_index,:]
			dev_y, val_y = train_y[dev_index], train_y[val_index]

			if model_name == "XGB1":
				pred_val, loss, pred_test = runXGB(dev_X, dev_y, val_X, val_y, test_X, rounds=5000, dep=8, feature_names=dev_X.columns.tolist())
			elif model_name == "LGB1":
				pred_val, loss, pred_test = runLGB(dev_X, dev_y, val_X, val_y, test_X, rounds=5000, dep=8)
			pred_val_full[val_index] = pred_val
			pred_test_full = pred_test_full + pred_test
			cv_scores.append(loss)
			print cv_scores
		pred_test_full /= 5.
		print metrics.roc_auc_score(train_y, pred_val_full)

		out_df = pd.DataFrame({"UCIC_ID":test_id})
		out_df["Responders"] = pred_test_full
		out_df.to_csv("./meta_models/test/pred_test_v5_"+model_name+".csv", index=False)

		out_df = pd.DataFrame({"UCIC_ID":train_id})
		out_df["Responders"] = pred_val_full
		out_df.to_csv("./meta_models/val/pred_val_v5_"+model_name+".csv", index=False)
