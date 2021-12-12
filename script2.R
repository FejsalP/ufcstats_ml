library(dplyr)
df <- read.csv("ufcstats_cleaned.csv", header = TRUE, sep = ",")

str(df) 
set.seed(1)

#Boxplots

control_times <- select(df, c(ctrl_time_rnd_1, ctrl_time_rnd_2, ctrl_time_rnd_3))
boxplot(control_times)

boxplot(winner ~ ctrl_time_rnd_1, data=df, xlab='Time', ylab='Winner')
# Merging red and blue columns
df <- df %>% mutate (kd_rnd_1 = red_kd_rnd_1 - blue_kd_rnd_1)
df <- df %>% mutate (sig_str_rnd_1 = red_sig_str_rnd_1 - blue_sig_str_rnd_1)
df <- df %>% mutate (ctrl_time_rnd_1 = red_ctrl_time_rnd_1 - blue_ctrl_time_rnd_1)
df <- df %>% mutate (ctrl_time_rnd_2 = red_ctrl_time_rnd_2 - blue_ctrl_time_rnd_2)
df <- df %>% mutate (ctrl_time_rnd_3 = red_ctrl_time_rnd_3 - blue_ctrl_time_rnd_3)

# ..
df$winner <- ifelse(df$winner=='red', 1, 0)

# Splitting on training and testing data
# 80% on training, 20% on test
length(df$winner)
0.8*2675
sample_ <- sample(2675, 2140)

training_data <- df[sample_,]
test_data <- df[-sample_,]

prop.table(table(training_data$winner)) #0.37 blue, 0627 red
prop.table(table(test_data$winner)) #0.4 blue, 0.6 red
# Proportion of the target class is almost the same, so we are good to go.

# Training the data on the C5.0 decision tree
# Importing C5.0 decision tree
library(C50)
x <- training_data[-1] # All features except class.
y <- as.factor(training_data$winner)
ufc_dtree <- C5.0(x, y)

# Plotting decision tree
plot(ufc_dtree)

# Checking attribute usage
summary(ufc_dtree) # Several attributes can be removed.
# Control time, Total Strikes, Significant Strikes
# Significant Strikes Head, Clinch and Ground, Takedowns, 
# Significant Strikes Head, Body and Ground attempt
# Total strike attempt
# 

## Evaluating performance

# Creating predictions
ufc_predictions_dtree <- predict(object = ufc_dtree, test_data)
ufc_predictions
test_data$winner

library(gmodels)
#confusion matrix #1, absence is "positive" class
CrossTable(test_data$winner, ufc_predictions_dtree, 
           prop.chisq = FALSE, dnn=c('Actual', 'Predicted'))
library(caret)
#confusion matrix # positive is 1 (absence)
confusionMatrixDTree <- confusionMatrix(as.factor(ufc_predictions_dtree), 
                                    as.factor(test_data$winner), 
                                    positive = "1", 
                                    dnn= c("Predicted", "Reference/Actual"))
dtree_metrics <- c(confusionMatrixDTree$overall['Accuracy'], 
                   confusionMatrixDTree$overall['Kappa'], 
                   confusionMatrixDTree$byClass['Sensitivity'], 
                   confusionMatrixDTree$byClass['Specificity'])
dtree_metrics

#5 Improving the performance of the model

# Using boosting

# Random forests
library(randomForest)

# x - training data
# y - factor vector with class for each row in the training data

ufc_rf <- randomForest(x, y, ntree = 500, mtry = sqrt(length(x)), importance=TRUE)

ufc_predictions_rf <- predict(ufc_rf, test_data, type='response')

# Evaluating performance

# Confusion matrix for Random Forest
confusionMatrix_rf <- confusionMatrix(as.factor(ufc_predictions_rf), 
                                      as.factor(test_data$winner), 
                                      positive = "1", 
                                      dnn= c("Predicted", "Reference/Actual"))
confusionMatrix_rf
rf_metrics <- c(confusionMatrix_rf$overall['Accuracy'], 
                confusionMatrix_rf$overall['Kappa'], 
                confusionMatrix_rf$byClass['Sensitivity'], 
                confusionMatrix_rf$byClass['Specificity'])

# Cross-validation 10-fold for C5.0 decision tree
library(caret)
library(MLmetrics)

#twoClassSummary for Sensitivity, ROC and Specificity
#prSummary for  AUC, Precision, Recall and F
#defaultSummary Accuracy and Kappa

mixSummary <- function(data, lev = NULL, model = NULL){
  out <- c(twoClassSummary(data, lev, model), prSummary(data, lev, model), defaultSummary(data, lev, model))
  return (out)
}

trControl <- trainControl(method='cv',
                          number = 10,
                          classProbs = TRUE,
                          summaryFunction = mixSummary,
                          savePredictions = TRUE)
training_data$winner <- ifelse(training_data$winner==1, 'red', 'blue')
ufc_dtree_cv_10 <- train (x = training_data[,-1],
                          y = training_data[,1],
                          data = training_data, 
                          method ='C5.0', 
                          trControl = trControl, 
                          metric = 'Accuracy',)
ufc_dtree_cv_10$results[c('Accuracy', 'AUC', 'Sens', 'Spec')]
View(ufc_dtree_cv_10)

pred_dtree_cv <- predict(ufc_dtree_cv_10, newdata = test_data)
pred_dtree_cv
test_data$winner <- ifelse(test_data$winner==1, 'red', 'blue')
confusionMatrix_dtree_cv <- confusionMatrix(as.factor(as.factor(pred_dtree_cv)), 
                                                      as.factor(test_data$winner), 
                                                      positive = "red", 
                                                      dnn= c("Predicted", "Reference/Actual"))
confusionMatrix_dtree_cv
dtree_cv_metrics <- c(confusionMatrix_dtree_cv$overall['Accuracy'], 
                      confusionMatrix_dtree_cv$overall['Kappa'], 
                      confusionMatrix_dtree_cv$byClass['Sensitivity'], 
                      confusionMatrix_dtree_cv$byClass['Specificity'])

# Cross-validation 10-fold for random forest

ufc_rf_cv_10 <- train (x = training_data[,-1],
                          y = training_data[,1],
                          data = training_data, 
                          method ='rf', 
                          trControl = trControl, 
                          metric = 'Accuracy',)
ufc_rf_cv_10$results[c('Accuracy', 'AUC', 'Sens', 'Spec')]
View(ufc_rf_cv_10)

pred_rf_cv <- predict(ufc_rf_cv_10, newdata = test_data)
pred_rf_cv
confusionMatrix_rf_cv <- confusionMatrix(as.factor(as.factor(pred_rf_cv)), 
                                            as.factor(test_data$winner), 
                                            positive = "red", 
                                            dnn= c("Predicted", "Reference/Actual"))
confusionMatrix_rf_cv
rf_cv_metrics <- c(confusionMatrix_rf_cv$overall['Accuracy'], 
                   confusionMatrix_rf_cv$overall['Kappa'], 
                   confusionMatrix_rf_cv$byClass['Sensitivity'], 
                   confusionMatrix_rf_cv$byClass['Specificity'])



all_metrics <- data.frame(dtree_metrics, rf_metrics, dtree_cv_metrics, rf_cv_metrics)
all_metrics # Increase in all 4 metrics after CV or RandomForests
# RF with 10CV takes ~10x more time to process and still delivers the same result
# 

################################################################################
####################################   KNN   ###################################
################################################################################

trControl_knn <- trainControl(method='cv',
                          number = 10,
                          classProbs = TRUE,
                          summaryFunction = mixSummary)

ufc_knn_cv_10 <- train (winner ~ .,
                          data = training_data, 
                          method ="knn", 
                          trControl = trControl, 
                          preProcess = c("scale"),
                          tuneGrid = expand.grid(k = 1:15),
                          metric = 'Accuracy',)







