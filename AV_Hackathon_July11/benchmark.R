## setting the working directory ##
setwd("../Data/")

## reading the train and test files ##
train = read.csv("train.csv")
test = read.csv("test.csv")

## removing the categorical columns for benchmark script. Create dummy variables for further improvement ##
test_id = test["id"]
train["id"] = NULL
test["id"] = NULL
train["Category_article"] = NULL
test["Category_article"] = NULL
train["Day_of_publishing"] = NULL
test["Day_of_publishing"] = NULL

## creating a linear regression model and predicting on teset set ##
## change the modeling methodology and try different models ##
model = lm(shares~., data=train)
summary(model)
preds = predict(model, test, type='response')

## writing the outputs to csv file ##
out_df = data.frame(test_id, preds)
names(out_df) = c("id", "predictions")
write.csv(out_df, "benchmark_R.csv", row.names=F, quote=F)
