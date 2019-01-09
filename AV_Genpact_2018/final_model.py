import datetime
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, metrics, ensemble
import lightgbm as lgb

def runLGB(train_X, train_y, test_X, test_y=None, test_X2=None, dep=10, seed=0, rounds=20000): 
	params = {}
	params["objective"] = "regression"
	params['metric'] = 'rmse'
	params["max_depth"] = dep
	params["min_data_in_leaf"] = 100
	params["learning_rate"] = 0.04
	params["bagging_fraction"] = 0.7
	params["feature_fraction"] = 0.5
	params["bagging_freq"] = 5
	params["bagging_seed"] = seed
	#params["lambda_l2"] = 0.01
	params["verbosity"] = -1
	num_rounds = rounds

	plst = list(params.items())
	lgtrain = lgb.Dataset(train_X, label=train_y)

	if test_y is not None:
		lgtest = lgb.Dataset(test_X, label=test_y)
		model = lgb.train(params, lgtrain, num_rounds, valid_sets=[lgtest], early_stopping_rounds=200, verbose_eval=100)
	else:
		lgtest = lgb.Dataset(test_X)
		model = lgb.train(params, lgtrain, num_rounds)

	pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
	if test_X2 is not None:
		pred_test_y2 = model.predict(test_X2, num_iteration=model.best_iteration)
	imps = model.feature_importance()
	names = model.feature_name()
	for fi, fn in enumerate(names):
		print(fn, imps[fi])

	loss = 0
	if test_y is not None:
		loss = np.sqrt(metrics.mean_squared_error(test_y, pred_test_y))
		print(loss)
		return pred_test_y, loss, pred_test_y2, model.best_iteration
	else:
		return pred_test_y


def run_model(week_num):
	print("WEEK NUMBER IS : ", week_num)
	week_shift_map = {
	146 : ["target_shift1", "target_shift2", "target_shift3", "target_shift4", "target_shift5", "target_shift6", "target_shift7", "target_shift8", "target_shift9", "target_shift10", "target_shift11", "target_shift12", "target_shift13"],
	147 : ["target_shift2", "target_shift3", "target_shift4", "target_shift5", "target_shift6", "target_shift7", "target_shift8", "target_shift9", "target_shift10", "target_shift11", "target_shift12", "target_shift13"],
	148 : ["target_shift3", "target_shift4", "target_shift5", "target_shift6", "target_shift7", "target_shift8", "target_shift9", "target_shift10", "target_shift11", "target_shift12", "target_shift13"],
	149 : ["target_shift4", "target_shift5", "target_shift6", "target_shift7", "target_shift8", "target_shift9", "target_shift10", "target_shift11", "target_shift12", "target_shift13"],
	150 : ["target_shift5", "target_shift6", "target_shift7", "target_shift8", "target_shift9", "target_shift10", "target_shift11", "target_shift13"],
	151 : ["target_shift6", "target_shift7", "target_shift8", "target_shift9", "target_shift10", "target_shift11", "target_shift13"],
	152 : ["target_shift7", "target_shift8", "target_shift9", "target_shift10", "target_shift11", "target_shift13"],
	153 : ["target_shift8", "target_shift9", "target_shift10", "target_shift11", "target_shift13"],
	154 : ["target_shift9", "target_shift10", "target_shift11", "target_shift13"],
	155 : ["target_shift10", "target_shift11", "target_shift13"]
	}

	train_df = pd.read_csv("../input/train.csv")
	test_df = pd.read_csv("../input/test_QoiMO9B.csv")
	center_df = pd.read_csv("../input/fulfilment_center_info.csv")
	meal_df = pd.read_csv("../input/meal_info.csv")

	train_df = pd.merge(train_df, center_df, on="center_id", how="left")
	test_df = pd.merge(test_df, center_df, on="center_id", how="left")
	train_df = pd.merge(train_df, meal_df, on="meal_id", how="left")
	test_df = pd.merge(test_df, meal_df, on="meal_id", how="left")

	cat_cols = ["center_type", "category", "cuisine"]
	for c in cat_cols:
		lbl = preprocessing.LabelEncoder()
		lbl.fit(list(train_df[c].values.astype('str')) + list(test_df[c].values.astype('str')))
		train_df[c] = lbl.transform(list(train_df[c].values.astype('str')))
		test_df[c] = lbl.transform(list(test_df[c].values.astype('str')))

	train_df["discount_ratio"] = train_df["base_price"] / train_df["checkout_price"]
	test_df["discount_ratio"] = test_df["base_price"] / test_df["checkout_price"]

	train_df["train_set"] = 1
	test_df["train_set"] = 0
	test_df["num_orders"] = -99

	print(train_df.shape)
	all_df = pd.concat([train_df, test_df])
	all_df = all_df.sort_values(by=["center_id", "meal_id", "week"]).reset_index(drop=True)
	print(all_df.shape)
	all_df["target_shift1"] = all_df.groupby(["center_id", "meal_id"])["num_orders"].shift(1)
	all_df["target_shift2"] = all_df.groupby(["center_id", "meal_id"])["num_orders"].shift(2)
	all_df["target_shift3"] = all_df.groupby(["center_id", "meal_id"])["num_orders"].shift(3)
	all_df["target_shift4"] = all_df.groupby(["center_id", "meal_id"])["num_orders"].shift(4)
	all_df["target_shift5"] = all_df.groupby(["center_id", "meal_id"])["num_orders"].shift(5)
	all_df["target_shift6"] = all_df.groupby(["center_id", "meal_id"])["num_orders"].shift(6)
	all_df["target_shift7"] = all_df.groupby(["center_id", "meal_id"])["num_orders"].shift(7)
	all_df["target_shift8"] = all_df.groupby(["center_id", "meal_id"])["num_orders"].shift(8)
	all_df["target_shift9"] = all_df.groupby(["center_id", "meal_id"])["num_orders"].shift(9)
	all_df["target_shift10"] = all_df.groupby(["center_id", "meal_id"])["num_orders"].shift(10)
	all_df["target_shift11"] = all_df.groupby(["center_id", "meal_id"])["num_orders"].shift(11)
	all_df["target_shift12"] = all_df.groupby(["center_id", "meal_id"])["num_orders"].shift(12)
	all_df["target_shift13"] = all_df.groupby(["center_id", "meal_id"])["num_orders"].shift(13)

	all_df["discount_shift1"] = all_df.groupby(["center_id", "meal_id"])["discount_ratio"].shift(1)
	all_df["discount_shift2"] = all_df.groupby(["center_id", "meal_id"])["discount_ratio"].shift(2)
	all_df["discount_shift3"] = all_df.groupby(["center_id", "meal_id"])["discount_ratio"].shift(3)

	#### center shift features ###
	#gdf = all_df.groupby(["center_id", "category", "week"])["target_shift11"].agg(['sum']).reset_index()
	#gdf.columns = ["center_id", "category", "week", "center_week_orders11"]
	#all_df = all_df.merge(gdf, on=["center_id", "category", "week"], how="left")
	gdf = all_df.groupby(["category"])["id"].agg(['size']).reset_index()
	gdf.columns = ["category", "cat_count"]
	all_df = all_df.merge(gdf, on=["category"], how="left")

	gdf = all_df.groupby(["cuisine"])["id"].agg(['size']).reset_index()
	gdf.columns = ["cuisine", "cui_count"]
	all_df = all_df.merge(gdf, on=["cuisine"], how="left")

	gdf = all_df.groupby(["city_code"])["id"].agg(['size']).reset_index()
	gdf.columns = ["city_code", "city_count"]
	all_df = all_df.merge(gdf, on=["city_code"], how="left")

	gdf = all_df.groupby(["region_code"])["id"].agg(['size']).reset_index()
	gdf.columns = ["region_code", "region_count"]
	all_df = all_df.merge(gdf, on=["region_code"], how="left")

	#gdf = all_df.groupby(["city_code", "category"])["id"].agg(['size']).reset_index()
	#gdf.columns = ["city_code", "category", "city_cat_count"]
	#all_df = all_df.merge(gdf, on=["city_code", "category"], how="left")

	#gdf = all_df.groupby(["city_code", "cuisine"])["id"].agg(['size']).reset_index()
	#gdf.columns = ["city_code", "cuisine", "city_cui_count"]
	#all_df = all_df.merge(gdf, on=["city_code", "cuisine"], how="left")

	#gdf = all_df.groupby(["region_code", "category"])["id"].agg(['size']).reset_index()
	#gdf.columns = ["region_code", "category", "region_cat_count"]
	#all_df = all_df.merge(gdf, on=["region_code", "category"], how="left")

	#gdf = all_df.groupby(["region_code", "cuisine"])["id"].agg(['size']).reset_index()
	#gdf.columns = ["region_code", "cuisine", "region_cui_count"]
	#all_df = all_df.merge(gdf, on=["region_code", "cuisine"], how="left")

	### Center count features ###
	gdf = all_df.groupby(["center_id", "week"])["id"].agg(['size']).reset_index()
	gdf.columns = ["center_id", "week", "center_week_count"]
	all_df = all_df.merge(gdf, on=["center_id", "week"], how="left")

	gdf = all_df.groupby(["center_id", "category"])["id"].count().reset_index()
	gdf.columns = ["center_id", "category", "center_cat_count"]
	all_df = all_df.merge(gdf, on=["center_id", "category"], how="left")

	gdf = all_df.groupby(["center_id", "category", "week"])["id"].count().reset_index()
	gdf.columns = ["center_id", "category", "week", "center_cat_week_count"]
	#gdf = gdf.sort_values(by=["center_id", "category", "week"]).reset_index(drop=True)
	#gdf["center_cat_week1_count"] = gdf.groupby(["center_id", "category", "week"])["center_cat_week_count"].shift(1)
	all_df = all_df.merge(gdf, on=["center_id", "category", "week"], how="left")

	gdf = all_df.groupby(["center_id", "cuisine"])["id"].count().reset_index()
	gdf.columns = ["center_id", "cuisine", "center_cui_count"]
	all_df = all_df.merge(gdf, on=["center_id", "cuisine"], how="left")


	### Meal count features ###
	gdf = all_df.groupby(["meal_id"])["id"].count().reset_index()
	gdf.columns = ["meal_id", "meal_count"]
	all_df = all_df.merge(gdf, on=["meal_id"], how="left")

	gdf = all_df.groupby(["region_code", "meal_id"])["id"].count().reset_index()
	gdf.columns = ["region_code", "meal_id", "region_meal_count"]
	all_df = all_df.merge(gdf, on=["region_code", "meal_id"], how="left")

	gdf = all_df.groupby(["meal_id", "week"])["id"].count().reset_index()
	gdf.columns = ["meal_id", "week", "meal_week_count"]
	all_df = all_df.merge(gdf, on=["meal_id", "week"], how="left")

	gdf = all_df.groupby(["center_type", "meal_id", "week"])["id"].count().reset_index()
	gdf.columns = ["center_type", "meal_id", "week", "type_meal_week_count"]
	all_df = all_df.merge(gdf, on=["center_type", "meal_id", "week"], how="left")

	gdf = all_df.groupby(["region_code", "meal_id", "week"])["id"].count().reset_index()
	gdf.columns = ["region_code", "meal_id", "week", "region_meal_week_count"]
	all_df = all_df.merge(gdf, on=["region_code", "meal_id", "week"], how="left")

	gdf = all_df.groupby(["city_code", "meal_id", "week"])["id"].count().reset_index()
	gdf.columns = ["city_code", "meal_id", "week", "city_meal_week_count"]
	all_df = all_df.merge(gdf, on=["city_code", "meal_id", "week"], how="left")

	### Price rank ###
	all_df["meal_price_rank"] = all_df.groupby("meal_id")["checkout_price"].rank()
	all_df["meal_city_price_rank"] = all_df.groupby(["meal_id", "city_code"])["checkout_price"].rank()
	all_df["meal_region_price_rank"] = all_df.groupby(["meal_id", "region_code"])["checkout_price"].rank()
	all_df["meal_week_price_rank"] = all_df.groupby(["meal_id", "week"])["checkout_price"].rank()

	all_df["center_price_rank"] = all_df.groupby("center_id")["checkout_price"].rank()
	all_df["center_week_price_rank"] = all_df.groupby(["center_id", "week"])["checkout_price"].rank()
	all_df["center_cat_price_rank"] = all_df.groupby(["center_id", "category"])["checkout_price"].rank()

	### Week features ###
	gdf = all_df.groupby(["meal_id"])["checkout_price"].agg(["min", "max", "mean", "std"]).reset_index()
	gdf.columns = ["meal_id", "meal_price_min", "meal_price_max", "meal_price_mean", "meal_price_std"]
	all_df = all_df.merge(gdf, on=["meal_id"], how="left")

	gdf = all_df.groupby(["meal_id"])["base_price"].agg(["min", "max", "mean", "std"]).reset_index()
	gdf.columns = ["meal_id", "disc_price_min", "disc_price_max", "disc_price_mean", "disc_price_std"]
	all_df = all_df.merge(gdf, on=["meal_id"], how="left")

	gdf = all_df.groupby(["city_code","meal_id", "week"])["checkout_price"].agg(["min", "max", "mean", "std"]).reset_index()
	gdf.columns = ["city_code", "meal_id", "week", "meal_price2_min", "meal_price2_max", "meal_price2_mean", "meal_price2_std"]
	all_df = all_df.merge(gdf, on=["city_code", "meal_id", "week"], how="left")

	gdf = all_df.groupby(["city_code", "category"])["checkout_price"].agg(["mean", "std"]).reset_index()
	gdf.columns = ["city_code", "category", "meal_price3_mean", "meal_price3_std"]
	all_df = all_df.merge(gdf, on=["city_code", "category"], how="left")

	#gdf = all_df.groupby(["region_code","meal_id", "week"])["checkout_price"].agg(["min", "max", "mean", "std"]).reset_index()
	#gdf.columns = ["region_code", "meal_id", "week", "meal_price4_min", "meal_price4_max", "meal_price4_mean", "meal_price4_std"]
	#all_df = all_df.merge(gdf, on=["region_code", "meal_id", "week"], how="left")


	### New ones ###
	#all_df["ratio1"] = all_df["target_shift10"] / all_df["op_area"]	
	#all_df["ratio2"] = all_df["target_shift10"] / all_df["checkout_price"]	

	### overall mean sum ###
	#gdf = all_df.groupby(["meal_id", "week"])["target_shift10"].sum().reset_index()
	#gdf.columns = ["meal_id", "week", "city_meal_week_lag10"]
	#all_df = all_df.merge(gdf, on=["meal_id", "week"], how="left")

	train_df = all_df[all_df["train_set"]==1].reset_index(drop=True)
	test_df = all_df[all_df["train_set"]==0].reset_index(drop=True)
	test_df = test_df[test_df["week"] == week_num].reset_index(drop=True)

	dev_df = train_df[train_df["week"]<=135]	
	#dev_df = dev_df[dev_df["week"]>20]	
	val_df = train_df[train_df["week"]>135]
	train_y = np.log1p(train_df["num_orders"].values)
	dev_y = np.log1p(dev_df["num_orders"].values)
	val_y = np.log1p(val_df["num_orders"].values)
	cols_to_use = ["center_id", "meal_id", "checkout_price", "base_price", "discount_ratio", "emailer_for_promotion", "homepage_featured"]
	cols_to_use += ["city_code","region_code","center_type","op_area"]
	cols_to_use += ["category", "cuisine"]
	cols_to_use += ["cat_count", "cui_count", "city_count", "region_count"]
	#cols_to_use += ["city_cat_count", "city_cui_count", "region_cat_count", "region_cui_count"]
	cols_to_use += ["center_cat_count", "center_cui_count", "center_week_count"]
	cols_to_use += ["meal_week_count", "type_meal_week_count", "region_meal_week_count", "city_meal_week_count", "meal_count", "region_meal_count"]
	cols_to_use += ["meal_price_rank", "meal_city_price_rank", "meal_region_price_rank", "meal_week_price_rank"]
	cols_to_use += ["center_price_rank", "center_cat_price_rank", "center_week_price_rank"]
	cols_to_use += ["meal_price_min", "meal_price_max", "meal_price_mean", "meal_price_std"]
	cols_to_use += ["disc_price_min", "disc_price_max", "disc_price_mean", "disc_price_std"]
	cols_to_use += ["meal_price2_min", "meal_price2_max", "meal_price2_mean", "meal_price2_std"]
	cols_to_use += ["meal_price3_mean", "meal_price3_std"]
	cols_to_use += week_shift_map[week_num]

	train_X = train_df[cols_to_use]
	dev_X = dev_df[cols_to_use]
	val_X = val_df[cols_to_use]
	test_X = test_df[cols_to_use]
	print(val_X.tail())

	pred_val, loss, pred_test, nrounds = runLGB(dev_X, dev_y, val_X, val_y, test_X)
	pred_test1 = runLGB(train_X, train_y, test_X, rounds=nrounds)
	pred_test2 = runLGB(train_X, train_y, test_X, rounds=nrounds, seed=2018)
	pred_test = 0.5*pred_test1 + 0.5*pred_test2

	test_id = list(test_df["id"].values)
	test_preds = list(np.expm1(pred_test))
	return test_id, test_preds, loss

if __name__ == "__main__":
	test_ids = [] 
	preds = []
	cv = []
	for week_num in [146, 147, 148, 149, 150, 151, 152, 153, 154, 155]:
		ids, prs, ll = run_model(week_num)
		test_ids.extend(ids)
		preds.extend(prs)
		cv.append(ll)
		print(cv)
	sub_df = pd.DataFrame({"id":test_ids})
	sub_df["num_orders"] = preds
	sub_df.to_csv("sub8.csv", index=False)
