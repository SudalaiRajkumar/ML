import csv
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder

def getFeatures(df, dv_list=set(), start_date=datetime.datetime(2006,1,1)):
	grouped_df = df.groupby("Client_ID")
	for name, group in grouped_df:
		#if group.shape[0] < 2:
		#	continue
		#print name
		out = [name]
		#print group

		# time since last transaction #
		max_date = max(group["Transaction_Date"])
		out.append( (start_date - max_date).days )

		# Number of transactions #
		out.append(group.shape[0])

		# Mean EMI #
		out.append( np.mean(group["Number_of_EMI"]) )

		# Mean var1 #
		out.append( np.mean(group["Var1"]) )

		# Mean Var2 #
		out.append( np.mean(group["Var2"]) )

		# Mean Var3 #
                out.append( np.mean(group["Var3"]) )

		# Mean Transaction_Amount #
		out.append( np.mean(group["Transaction_Amount"]) )

		# Mean Purchased_in_Sale #
		out.append( np.mean(group["Purchased_in_Sale"]) )

		# get last purchase #
		last_purchase = group[group["Transaction_Date"] == max_date]
		#print "Last Purchase is : ", last_purchase

		# last purchase in sale #
		out.append( int(last_purchase["Purchased_in_Sale"].iloc[-1]) )

		# last EMI #
		out.append( int(last_purchase["Number_of_EMI"].iloc[-1]) )

		# last store #
		out.append( int(last_purchase["Store_ID"].iloc[-1]) )

		# last var1 #
                out.append( int(last_purchase["Var1"].iloc[-1]) )

		# last var2 #
                out.append( int(last_purchase["Var2"].iloc[-1]) )

		# last var3 #
                out.append( int(last_purchase["Var3"].iloc[-1]) )

		# Gender #
                out.append( int(last_purchase["Gender"].iloc[-1]) )

		# Last Referred_Friend #
		out.append( int(last_purchase["Referred_Friend"].iloc[-1]) )

		# Last SE category # 
		out.append( int(last_purchase["Sales_Executive_Category"].iloc[-1]) )

		# Last SE ID #
		out.append( int(last_purchase["Sales_Executive_ID"].iloc[-1]) )
		
		# Last Lead Source #
		out.append( int(last_purchase["Lead_Source_Category"].iloc[-1]) )

		# Last Payment Mode #
		out.append( int(last_purchase["Payment_Mode"].iloc[-1]) )

		# last product category #
		out.append( int(last_purchase["Product_Category"].iloc[-1]) )

		# last transaction amount #
		out.append( int(last_purchase["Transaction_Amount"].iloc[-1]) )

		# time since first transaction #
		min_date = min(group["Transaction_Date"])
                out.append( (start_date - min_date).days )

		# total time #
		out.append((max_date - min_date).days)

		# frequency #
		out.append( (max_date - min_date).days / float(group.shape[0]) )

		# number of unique stores visited #
		out.append( len( np.unique( group["Store_ID"] )) )

		# number of unique purchased in sale #
                out.append( len( np.unique( group["Purchased_in_Sale"] )) )

		# number of unique var1 #
		out.append( len( np.unique( group["Var1"] )) )

		# number of unique var2 #
                out.append( len( np.unique( group["Var2"] )) )

		# number of unique var3 #
                out.append( len( np.unique( group["Var3"] )) )

		# number of unique SE id #
		out.append( len( np.unique( group["Sales_Executive_ID"] )) )
	
		# number of unique SE cat #
		out.append( len( np.unique( group["Sales_Executive_Category"] )) )

		# number of unique LS cat #
                out.append( len( np.unique( group["Lead_Source_Category"] )) )
		
		# number of unique paymenr mode #
		out.append( len( np.unique( group["Payment_Mode"])) )

		# number of unique product category #
		out.append( len( np.unique( group["Product_Category"])) )
	
		# getting year of birth #
                yob = int((last_purchase["DOB"].iloc[-1]).year)
		if yob > 2000:
			yob = yob-100
		out.append(yob)	

		# number of unique dob #
                out.append( len( np.unique( group["DOB"])) )		

		# number of purchases in last one year #
		yop = (start_date.year - 1)
		temp_arr = np.array( group["Transaction_Date"].apply(lambda x: int(x.year>=yop)) )
		out.append(sum(temp_arr))
		out.append( np.sum( np.array(group["Transaction_Amount"]) * temp_arr ) )

		# number of purchases in last two years #
                yop = (start_date.year - 2)
                temp_arr = np.array( group["Transaction_Date"].apply(lambda x: int(x.year>=yop)) )
                out.append(sum(temp_arr))
		out.append( np.sum( np.array(group["Transaction_Amount"]) * temp_arr ) )

		# number of purchases in last three years #
                yop = (start_date.year - 3)
                temp_arr = np.array( group["Transaction_Date"].apply(lambda x: int(x.year>=yop)) )
                out.append(sum(temp_arr))
		out.append( np.sum( np.array(group["Transaction_Amount"]) * temp_arr ) )

		# DV #
		if name in dv_list:
			out.append(1)
		else:
			out.append(0)

		yield out

if __name__ == "__main__":
	train = pd.read_csv("../Data/dev.csv")
	repeat_clients = set(np.unique(pd.read_csv("../Data/val.csv")["Client_ID"]))
	print len(repeat_clients)
	test = pd.read_csv("../Data/Train_seers_accuracy.csv")

	print "Label Encoding.."
        for var in test.columns:
                if test[var].dtypes == object :
			if var in ["Transaction_Date", "DOB"]:
				continue
                        print var
                        lb = LabelEncoder()
                        full_var_data = pd.concat((train[var],test[var]),axis=0).astype('str')
                        lb.fit(np.array(full_var_data))
                        train[var] = lb.transform(np.array( train[var] ).astype('str'))
                        test[var] = lb.transform(np.array( test[var] ).astype('str'))

	train["Transaction_Date"] = pd.to_datetime(train["Transaction_Date"], format="%d-%b-%y")
	test["Transaction_Date"] = pd.to_datetime(test["Transaction_Date"], format="%d-%b-%y")
	print min(train["Transaction_Date"])
	print max(train["Transaction_Date"])
	train["DOB"] = pd.to_datetime(train["DOB"], format="%d-%b-%y")
        test["DOB"] = pd.to_datetime(test["DOB"], format="%d-%b-%y")
        print min(train["DOB"])
        print max(train["DOB"])

	print "Processing train.."
	out_file = open("train_features3.csv","w")
	writer = csv.writer(out_file)
	header = ["Client_ID", "TimeSinceLastTrans", "NumberOfTrans", "MeanEMI", "MeanVar1", "MeanVar2", "MeanVar3", "MeanTransactionAmount", "MeanPurchasedInSale", "LastPurchasedInSale", "LastEMI", "LastStoreID", "LastVar1", "LastVar2", "LastVar3", "Gender", "LastReferredFriend", "LastSECat", "LastSEID", "LastLeadSource", "LastPayMode", "LastProdCat", "LastTransAmt", "TimeSinceFirstTrans", "TotalTime", "FreqTrans", "NumUniqueStore", "NumUniPurchasedInSale", "NumUniVar1", "NumUniVar2", "NumUniVar3", "NumUniSEID", "NumUniSECat", "NumUniLScat", "NumUniPayMode", "NumUniProdCat", "YoB", "NumUniDOB", "Last1YCount", "Last1YTA", "Last2YCount", "Last2YTA", "Last3YCount", "Last3YTA", "DV"]
	len_header = len(header)
	writer.writerow(header)
	count = 0
	for feature_list in getFeatures(train, repeat_clients, start_date=datetime.datetime(2006,1,1)):
		assert len_header == len(feature_list)
		writer.writerow( feature_list )
		#break
		count +=1
		if count%10000 == 0:
			print count
	out_file.close()

	print "Processing test..."
	out_file = open("test_features3.csv","w")
        writer = csv.writer(out_file)
	#header = ["Client_ID", "TimeSinceLastTrans", "NumberOfTrans", "MeanEMI", "MeanVar1", "MeanVar2", "MeanVar3", "MeanTransactionAmount", "MeanPurchasedInSale", "LastPurchasedInSale", "DV"]
	#len_header = len(header)
        writer.writerow(header)
	count = 0
        for feature_list in getFeatures(test, start_date=datetime.datetime(2007,1,1)):
		assert len_header == len(feature_list)
                writer.writerow( feature_list )
		count += 1
		if count%10000 == 0:
                        print count
        out_file.close()
