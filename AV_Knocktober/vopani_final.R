## setting working directory
path <- "/Volumes/External SD/AnalyticsVidhya/Knocktober"
setwd(path)

seed <- 235
set.seed(seed)


## loading libraries
library(data.table)
library(xgboost)


## loading data
train <- fread("./raw/Train.csv")
test <- fread("./raw/Test_D7W1juQ.csv")

health_camp <- fread("./raw/Health_Camp_Detail.csv")

health_1 <- fread("./raw/First_Health_Camp_Attended.csv")
health_2 <- fread("./raw/Second_Health_Camp_Attended.csv")
health_3 <- fread("./raw/Third_Health_Camp_Attended.csv")

health_1[, V5 := NULL]
setnames(health_1, "Health_Score", "Health_Score_1")
setnames(health_2, "Health Score", "Health_Score_2")

patient <- fread("./raw/Patient_Profile.csv")

train[, train_flag := 1]
test[, train_flag := 0]


## processing data
X_panel <- rbind(train, test)

X_panel <- merge(X_panel, health_1, all.x = TRUE, by = c("Patient_ID", "Health_Camp_ID"))
X_panel <- merge(X_panel, health_2, all.x = TRUE, by = c("Patient_ID", "Health_Camp_ID"))
X_panel <- merge(X_panel, health_3, all.x = TRUE, by = c("Patient_ID", "Health_Camp_ID"))

X_panel <- merge(X_panel, health_camp, all.x = TRUE, by = "Health_Camp_ID")
X_panel <- merge(X_panel, patient, all.x = TRUE, by = "Patient_ID")

X_panel[, target := 0]

X_panel$target[X_panel$Category1 != "Third" & (X_panel$Health_Score_1 > 0 | X_panel$Health_Score_2 > 0)] <- 1
X_panel$target[X_panel$Category1 == "Third" & X_panel$Number_of_stall_visited > 0] <- 1

X_panel[, ":="(Registration_Date = as.Date(Registration_Date, "%d-%b-%y"),
               Camp_Start_Date = as.Date(Camp_Start_Date, "%d-%b-%y"),
               Camp_End_Date = as.Date(Camp_End_Date, "%d-%b-%y"),
               First_Interaction = as.Date(First_Interaction, "%d-%b-%y"),
               Category1 = as.numeric(as.factor(Category1)),
               Category2 = as.numeric(as.factor(Category2)),
               City_Type = as.numeric(as.factor(City_Type)),
               Income = as.numeric(as.factor(Income)),
               Employer_Category = as.numeric(as.factor(Employer_Category)),
               Education_Score = as.numeric(Education_Score),
               Age = as.numeric(Age))]

setorder(X_panel, Patient_ID, Registration_Date)
X_panel$order <- seq(1, nrow(X_panel))

X_date <- X_panel[, c("Patient_ID", "Registration_Date", "order"), with = FALSE]
X_date$order <- X_date$order + 1
names(X_date)[2] <- "Prev_Date"

X_panel <- merge(X_panel, X_date, all.x = TRUE, by = c("Patient_ID", "order"))

X_date$order <- X_date$order - 2
names(X_date)[2] <- "Next_Date"

X_panel <- merge(X_panel, X_date, all.x = TRUE, by = c("Patient_ID", "order"))

X_panel[, ":="(Start_Date_Diff = as.numeric(Registration_Date - Camp_Start_Date),
               End_Date_Diff = as.numeric(Camp_End_Date - Registration_Date),
               Interaction_Date_Diff = as.numeric(Registration_Date - First_Interaction),
               Prev_Date_Diff = as.numeric(Registration_Date - Prev_Date),
               Next_Date_Diff = as.numeric(Registration_Date - Next_Date),
               Camp_Start_Year = year(Camp_Start_Date),
               Registration_Year = year(Registration_Date),
               Registration_Month = month(Registration_Date),
               Registration_Day = wday(Registration_Date))]

X_panel <- X_panel[Camp_Start_Year >= 2005]
X_panel <- X_panel[!is.na(Registration_Date)]
X_panel <- X_panel[Category3 == 2]

X_patient <- X_panel[, .(Count_Patient = .N), .(Patient_ID)]
X_panel <- merge(X_panel, X_patient, by = "Patient_ID")

X_patient_date <- X_panel[, .(Count_Patient_Date = .N), .(Patient_ID, Registration_Date)]
X_panel <- merge(X_panel, X_patient_date, by = c("Patient_ID", "Registration_Date"))

X_donation <- X_panel[Donation > 0, .(Min_Date_Donation = min(Registration_Date)), .(Patient_ID)]
X_panel <- merge(X_panel, X_donation, all.x = T, by = "Patient_ID")

X_panel[, Donation_Flag := ifelse(is.na(Min_Date_Donation), 0, ifelse(Registration_Date > Min_Date_Donation, 1, 0))]

X_train <- X_panel[train_flag == 1]
X_test <- X_panel[train_flag == 0]

X_features <- c("Count_Patient", "Count_Patient_Date", "Donation_Flag",
                "City_Type", "Income", "Education_Score", "Age",
                "Category1", "Category2",
                "Start_Date_Diff", "End_Date_Diff", "Prev_Date_Diff", "Next_Date_Diff")
X_target <- X_train$target

xgtrain <- xgb.DMatrix(data = as.matrix(X_train[, X_features, with = FALSE]), label = X_target, missing = NA)
xgtest <- xgb.DMatrix(data = as.matrix(X_test[, X_features, with = FALSE]), missing = NA)


## xgboost
params <- list()
params$objective <- "binary:logistic"
params$eta <- 0.1
params$max_depth <- 5
params$subsample <- 0.9
params$colsample_bytree <- 0.9
params$min_child_weight <- 2
params$eval_metric <- "auc"

model_xgb_cv <- xgb.cv(params=params, xgtrain, nrounds = 100, nfold = 5, early.stop.round = 30, prediction = TRUE)

model_xgb <- xgb.train(params = params, xgtrain, nrounds = 100)

vimp <- xgb.importance(model = model_xgb, feature_names = X_features)
View(vimp)


## submission
pred <- predict(model_xgb, xgtest)

submit <- data.table(Patient_ID = X_test$Patient_ID,
                     Health_Camp_ID = X_test$Health_Camp_ID,
                     Outcome = pred)

write.csv(submit, "./sub_vopani.csv", row.names = FALSE)
