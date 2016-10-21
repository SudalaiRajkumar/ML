import numpy as np
import pandas as pd

data_path = "../input/Train/"
first_camp = pd.read_csv( data_path + "First_Health_Camp_Attended.csv" )
second_camp = pd.read_csv( data_path + "Second_Health_Camp_Attended.csv" )
third_camp = pd.read_csv( data_path + "Third_Health_Camp_Attended.csv" )
print first_camp.shape, second_camp.shape, third_camp.shape

col_names = [['Patient_ID','Health_Camp_ID','Outcome']]
first_camp = first_camp[['Patient_ID','Health_Camp_ID','Health_Score']]
first_camp.columns = col_names
second_camp = second_camp[['Patient_ID','Health_Camp_ID','Health Score']]
second_camp.columns = col_names
third_camp = third_camp[['Patient_ID','Health_Camp_ID','Number_of_stall_visited']]
third_camp = third_camp[third_camp['Number_of_stall_visited']>0]
third_camp.columns = col_names
print third_camp.shape

all_camps = pd.concat([first_camp, second_camp, third_camp])
all_camps['Outcome'] = 1
print all_camps.shape

train = pd.read_csv(data_path + "Train.csv")
print train.shape

train = train.merge(all_camps, on=['Patient_ID','Health_Camp_ID'], how='left')
train['Outcome'] = train['Outcome'].fillna(0).astype('int')
train.to_csv(data_path+'train_with_outcome.csv', index=False)
print train.Outcome.value_counts()
