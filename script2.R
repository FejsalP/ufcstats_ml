library(dplyr)
library(caret)
library(C50)
library(gmodels)
library(caret)
library(MLmetrics)
library(randomForest)
library(class)
library(MLeval)
library(pROC)
library(ROCR)

df <- read.csv("ufcstats_cleaned.csv", header = TRUE, sep = ",")

str(df) 
set.seed(1)

#Boxplots

# control_times <- select(df, c(ctrl_time_rnd_1, ctrl_time_rnd_2, ctrl_time_rnd_3))
# boxplot(control_times)
# 
# boxplot(winner ~ ctrl_time_rnd_1, data=df, xlab='Time', ylab='Winner')
# # Merging red and blue columns
# df <- df %>% mutate (kd_rnd_1 = red_kd_rnd_1 - blue_kd_rnd_1)
# df <- df %>% mutate (sig_str_rnd_1 = red_sig_str_rnd_1 - blue_sig_str_rnd_1)
# df <- df %>% mutate (ctrl_time_rnd_1 = red_ctrl_time_rnd_1 - blue_ctrl_time_rnd_1)
# df <- df %>% mutate (ctrl_time_rnd_2 = red_ctrl_time_rnd_2 - blue_ctrl_time_rnd_2)
# df <- df %>% mutate (ctrl_time_rnd_3 = red_ctrl_time_rnd_3 - blue_ctrl_time_rnd_3)
# 
# # ..

# df$winner <- ifelse(df$winner=='red', 1, 0)

# Splitting on training and testing data
# 80% on training, 20% on test
length(df$winner) #number of samples in data
0.8*2675 # = 2140
sample_ <- sample(2675, 2140)

training_data <- df[sample_,]
test_data <- df[-sample_,]

prop.table(table(training_data$winner)) #0.37 blue, 0627 red
prop.table(table(test_data$winner)) #0.4 blue, 0.6 red
# Proportion of the target class is almost the same, so we are good to go.

# Training the data on the C5.0 decision tree
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

CrossTable(test_data$winner, ufc_predictions_dtree, 
           prop.chisq = FALSE, dnn=c('Actual', 'Predicted'))

confusionMatrixDTree <- confusionMatrix(as.factor(ufc_predictions_dtree), 
                                    as.factor(test_data$winner), 
                                    positive = "red", 
                                    dnn= c("Predicted", "Reference/Actual"))
dtree_metrics <- c(confusionMatrixDTree$overall['Accuracy'], 
                   confusionMatrixDTree$overall['Kappa'], 
                   confusionMatrixDTree$byClass['Sensitivity'], 
                   confusionMatrixDTree$byClass['Specificity'])
dtree_metrics

#5 Improving the performance of the model
# Using boosting


# Random forests
# x - training data
# y - factor vector with class for each row in the training data

ufc_rf <- randomForest(x, y, ntree = 500, mtry = sqrt(length(x)), importance=TRUE)

ufc_predictions_rf <- predict(ufc_rf, test_data, type='response')

# Evaluating performance

# Confusion matrix for Random Forest
confusionMatrix_rf <- confusionMatrix(as.factor(ufc_predictions_rf), 
                                      as.factor(test_data$winner), 
                                      positive = "red", 
                                      dnn= c("Predicted", "Reference/Actual"))
confusionMatrix_rf
rf_metrics <- c(confusionMatrix_rf$overall['Accuracy'], 
                confusionMatrix_rf$overall['Kappa'], 
                confusionMatrix_rf$byClass['Sensitivity'], 
                confusionMatrix_rf$byClass['Specificity'])

# Cross-validation 10-fold for C5.0 decision tree


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
# training_data$winner <- ifelse(training_data$winner==1, 'red', 'blue')
ufc_dtree_cv_10 <- train (x = training_data[,-1],
                          y = training_data[,1],
                          data = training_data, 
                          method ='C5.0', 
                          trControl = trControl, 
                          metric = 'ROC',)
ufc_dtree_cv_10$results[c('Accuracy', 'AUC', 'Sens', 'Spec')]
View(ufc_dtree_cv_10)

pred_dtree_cv <- predict(ufc_dtree_cv_10, newdata = test_data)
pred_dtree_cv
# test_data$winner <- ifelse(test_data$winner==1, 'red', 'blue')
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
                          metric = 'ROC',)
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




################################################################################
####################################   KNN   ###################################
################################################################################

#min_max normalization function
min_max_norm <- function(x){
  ((x-min(x))/(max(x)-min(x)))
}
#normalizing the data
ufc_normalized <- as.data.frame(lapply(df[,-1], min_max_norm))
training_data_knn <- ufc_normalized[sample_, ] #random 2140 rows
test_data_knn <- ufc_normalized[-sample_, ] # remaining 535 rows that are not in training_set (2675-2140)
training_label_knn <- df[sample_, 1] #target class for training data
test_label_knn <- df[-sample_, 1] #target class for test data

#training the data on KNN
sqrt(length(training_data$winner)) #rule of thumb for choosing K
ufc_knn <- knn(train = training_data_knn, test = test_data_knn, cl = training_label_knn, k = 45, prob=TRUE)

confusionMatrix_knn <-confusionMatrix(as.factor(ufc_knn),
                                as.factor(test_label_knn),
                                positive='red',
                                dnn= c("Predicted", "Reference/Actual"))

knn_metrics <- c(confusionMatrix_knn$overall['Accuracy'], 
                 confusionMatrix_knn$overall['Kappa'], 
                 confusionMatrix_knn$byClass['Sensitivity'], 
                 confusionMatrix_knn$byClass['Specificity'])


#train control object that is required for the train()
trControl_knn <- trainControl(method='cv',
                          number = 10,
                          classProbs = TRUE,
                          summaryFunction = mixSummary,
                          savePredictions = TRUE)
str(training_data)
training_data_test <- training_data
str(training_data_test)
training_data_test$winner <- as.factor(training_data_test$winner)
ufc_knn_cv_10 <- train (winner ~ .,
                          data = training_data, 
                          method ="knn", 
                          trControl = trControl_knn, 
                          preProcess = c("center", "scale"),
                          tuneGrid = expand.grid(k = 1:50),
                          metric = 'Accuracy')
evalm(ufc_knn_cv_10)
#Final value for k is 50
ufc_knn_cv_10$results[c('Accuracy', 'AUC', 'Sens', 'Spec')][50,] # Gets performance metrics when K=50


pred_knn_cv <- predict(ufc_knn_cv_10, newdata = test_data)

confusionMatrix_knn_cv <- confusionMatrix(as.factor(as.factor(pred_knn_cv)), 
                                         as.factor(test_data$winner), 
                                         positive = "red", 
                                         dnn= c("Predicted", "Reference/Actual"))

knn_cv_metrics <- c(confusionMatrix_knn_cv$overall['Accuracy'], 
                    confusionMatrix_knn_cv$overall['Kappa'], 
                    confusionMatrix_knn_cv$byClass['Sensitivity'], 
                    confusionMatrix_knn_cv$byClass['Specificity'])


all_metrics <- data.frame(dtree_metrics, rf_metrics, dtree_cv_metrics, rf_cv_metrics,knn_metrics, knn_cv_metrics)
all_metrics # Increase in all 4 metrics after CV or RandomForests
# RF with 10CV takes ~10x more time to process and still delivers the same result

#Plotting ROC curves for models with cross-validation (Decision tree, random forest and KNN) 
roc_curves <- evalm(list(ufc_dtree_cv_10, ufc_rf_cv_10, ufc_knn_cv_10),gnames=c('DTree CV10','RF CV10', 'KNN CV10'))
# AUC for DTree CV 10 = 0.9 roc_curves$stdres$`DTree CV10`$Score[13]
# AUC for RF CV 10 = 0.91 roc_curves$stdres$`RF CV10`$Score[13]
# AUC for KNN CV 10 = 0.88 roc_curves$stdres$`DTree CV10`$Score[13]
knn_cv_10_roc <- evalm(ufc_knn_cv_10, gnames='KNN CV10')

# test1 <- evalm(ufc_dtree_cv_10,plots='r',rlinethick=0.8,fsize=8)
# test2 <- evalm(ufc_rf_cv_10,plots='r',rlinethick=0.8,fsize=8)

## Getting AUC value for and plotting the ROC Curve for KNN (from class library)

prob_knn <- attr(ufc_knn, 'prob')
prob_knn <- 2*ifelse(ufc_knn == '-1', 1-prob, prob) - 1
pred_knn <- prediction(prob_knn, test_label_knn)
performance_knn <- performance(pred_knn, 'tpr','fpr')

#Plotting the ROC Curve for KNN
plot(performance_knn, avg='threshold',colorize=TRUE, lwd=3, main='ROC Curve')

#Getting the AUC value for KNN
knn_auc <- performance(pred_knn, measure='auc')
str(knn_auc)
knn_auc_value <- knn_auc@y.values #0.6926078 AUC for KNN

## Getting AUC value for and plotting the ROC Curve for C5.0 Decision Tree (from C5.0 library)

prob_dtree <- predict(ufc_dtree, test_data, type='prob')[,2] # only second column is needed i.e. probabilities for label 1/red
pred_dtree <- prediction(prob_dtree, labels = test_data$winner)
performance_dtree <- performance(pred_dtree, 'tpr','fpr')
#Plotting the ROC Curve for DTree C5.0
plot(performance_dtree, main='ROC Curve for Dtree')

#Getting the AUC value for DTree C5.0
dtree_auc <- performance(pred_dtree, measure='auc')
dtree_auc_value <- dtree_auc@y.values #0.8075596 AUC for Dtree
dtree_auc_value #0.8075596

# Getting AUC value and ploting the ROC Curve for Random Forest (from randomForest library)
prob_rf <- as.vector(ufc_rf$votes[,2]) #probabilites
pred_rf <- prediction(prob_rf, training_data$winner); #prediction instance

rf_auc <- performance(pred_rf, measure = "auc") # AUC performance
rf_auc_value <- rf_auc@y.values #AUC value
rf_auc_value
performance_rf <- performance(pred_rf, 'tpr','fpr') #true vs false positive performance
plot(performance_rf, main='ROC Curve for RF')

############################
########### SVM ############
############################

library(kernlab)

ufc_svm <- ksvm(winner ~ ., data = training_data_test, kernel='rbfdot')
pred_svm <- predict(ufc_svm, test_data)
head(pred_svm)
table(pred_svm)
agreement <- pred_svm == test_data$winner
table(agreement)
451/(451+84) #0.84 accuracy
453/(453+82)

