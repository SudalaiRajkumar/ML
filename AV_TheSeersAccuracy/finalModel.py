import csv
import pandas as pd
import numpy as np
import datetime
import operator
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn import ensemble
from sklearn.metrics import roc_auc_score,log_loss
import xgboost as xgb
import random

def create_feature_map(features):
        outfile = open('xgb.fmap', 'w')
        for i, feat in enumerate(features):
                outfile.write('{0}\t{1}\tq\n'.format(i,feat))
        outfile.close()

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, round_val=1650):
        params = {}
        params["objective"] = "binary:logistic"
        params['eval_metric'] = 'auc'
        params["eta"] = 0.01 
        params["min_child_weight"] = 2
        params["subsample"] = 0.55
        params["colsample_bytree"] = 0.9
        params["silent"] = 1
        params["max_depth"] = 4
        params["seed"] = seed_val
        params["max_delta_step"] = 2
        num_rounds = round_val

        plst = list(params.items())
        xgtrain = xgb.DMatrix(train_X, label=train_y)

        if test_y is not None:
                xgtest = xgb.DMatrix(test_X, label=test_y)
                watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
                model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=10000)
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
	print "Reading csv.."
	otrain = pd.read_csv("./train_features3.csv")
	otest = pd.read_csv("./test_features3.csv")
	print otrain.shape, otest.shape

	print "Getting DV.."
	train_y = np.array( otrain.DV.values )
	train_id = np.array( otrain.Client_ID.values )
	test_id = np.array( otest.Client_ID.values )

	print "Dropping.."
	otrain = otrain.drop(['DV'], axis=1)
        otest = otest.drop(["DV"], axis=1)

	use_cols = ['Client_ID', 'TimeSinceLastTrans', 'NumberOfTrans', 'MeanEMI', 'MeanVar1', 'MeanVar2', 'MeanVar3', 'MeanTransactionAmount', 'MeanPurchasedInSale', 'LastPurchasedInSale', 'LastEMI', 'LastStoreID', 'LastVar1', 'LastVar2', 'Gender', 'LastReferredFriend', 'LastSECat', 'LastSEID', 'LastLeadSource', 'LastPayMode', 'LastProdCat', 'LastTransAmt'] 
	train = otrain[use_cols]
	test = otest[use_cols]

	feat_names = list(train.columns)
        print "Converting to array.."
        train = np.array(train).astype('float')
        test = np.array(test).astype('float')
        print train.shape, test.shape

	assert train.shape[1] == test.shape[1]
	print "Final Model.."
	preds = runXGB(train, train_y, test, seed_val=0, round_val=1200)

	out_df = pd.DataFrame({"Client_ID":test_id})
	out_df["Cross_Sell"] = preds
	out_df.to_csv("submission.csv", index=False)
	

