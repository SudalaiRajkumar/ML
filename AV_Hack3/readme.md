##### Codes for Analytics Vidhya Online Hackathon 3.0 - Find the Next Brain Wong !

http://discuss.analyticsvidhya.com/t/online-hackathon-3-0-find-the-next-brain-wong/2838

###### My approach for the hackathon is as follows:

1. Converted all the categorical variables into one-hot encoded variables

2. Truncate the "Project Evaluation" value at 99.9th percentile value (value is 6121) - As Nalin mentioned in his post, if the DV distribution is different in the test set, then am done.

3. Built tree based models by selecting the params through cross validation

        a. Random Forest (2 models with different params - 1 with shorter trees and 1 with deep trees)

        b. Gradient Boosting (2 models with different params)

        c. Extreme Gradient Boosting (2 models with different params)

4. Simple weighted average of all the six models based on local validation

