Code for the Hackathon - [The Seers Accuracy](http://datahack.analyticsvidhya.com/contest/the-seers-accuracy) by [Analytics Vidhya](http://www.analyticsvidhya.com/)

####Objective
The objective of the competition is to predict whether the customer will come back in the next one year.

####Approach
We had transaction data of all the customers from Jan 2003 to Dec 2006. The idea is to predict whether the customers will come back in 2007 or not. 
1. The first step was to create a proper validation framework since there was no "target" variable
2. I have used transaction data from 2003 to 2005 to create the features. People who came back in 2006 were tagged as 1 and those who did not come back are tagged as 0, thereby getting the target column
3. Feature selection, models tuning were done using this validation sample.
4. For the final model, features were created using all the given data (2003 to 2006) and prediction was done for 2007.
5. People were using different types of approaches as well. [Vopani](https://github.com/rohanrao91/AnalyticsVidhya_SeersAccuracy) followed a two stage validation approach.

####Codes 
######splitDevVal.py
Code to split the data into development(2003 to 2005 data) and validation sample(2006 data) 

######createFeatures.py
Code to create the features from the given input dataset for both validation and final model

######finalModel.py
Code to get the final submission file
