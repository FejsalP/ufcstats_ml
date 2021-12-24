rm(list=ls())

# Importing necessary libraries
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
library(ggplot2)
library(jcolors)

df <- read.csv("ufcstats_rf_features.csv", header = TRUE, sep = ",")

str(df) 
set.seed(1)

################################################################################
################################################################################
################################################################################

# Splitting on training and testing data
# 80% on training, 20% on test
sample_ <- sample(length(df$winner), length(df$winner) * 0.8)

training_data <- df[sample_,]
test_data <- df[-sample_,]

#checking proportion of target class in training and test data
prop.table(table(training_data$winner)) 
prop.table(table(test_data$winner)) 


# all features except class attribute
training_data_without_class <- training_data[-1] 
# class attribute transformed into factor
training_data_only_class <- as.factor(training_data$winner)


################################################################################
###############################   FUNCTIONS    #################################
################################################################################

# function to create confusion matrix and return following metrics: accuracy,
# sensitivity, specificity
get_metrics <-function(model_predictions){
  confusionMatrix <- confusionMatrix(as.factor(model_predictions),
                                     as.factor(test_data$winner), 
                                     positive = "red",
                                     dnn= c("Predicted", "Reference/Actual"))
  
  metrics <- c(confusionMatrix$overall['Accuracy'],
               confusionMatrix$byClass['Sensitivity'],
               confusionMatrix$byClass['Specificity'])
  
  return (metrics)
}


# function used for trainControl summaryFunction. Returning vector
# with following metrics:
# twoClassSummary for Sensitivity, ROC and Specificity
# prSummary for  AUC, Precision, Recall and F
# defaultSummary Accuracy and Kappa
mix_summary <- function(data, lev = NULL, model = NULL){
  output <- c(twoClassSummary(data, lev, model), prSummary(data, lev, model), 
              defaultSummary(data, lev, model))
  # return (list(output[["AUC"]], output[["ROC"]], output[["Sens"]],
  #              output[["Spec"]], output[["Accuracy"]]))
  return(output)
}

# Getting the AUC value and performance object 
# that is required for plotting the ROC curve
get_auc_and_perf <- function (model){
  # only second column is needed i.e. probabilities for label 1/red
  probabilities <- predict(model, test_data, type='prob')[,2] 
  predictions <- prediction(probabilities, labels = test_data$winner)
  performance_ <- performance(predictions, 'tpr','fpr')
  
  auc <- performance(predictions, measure='auc')
  auc_value <- auc@y.values
  
  return (c(auc_value, performance_))
}

get_auc <- function (ufc_dtree_model){
  # getting AUC values
  prob_dtree <- predict(ufc_dtree_model, test_data, type='prob')[,2] # only second column is needed i.e. probabilities for label 1/red
  pred_dtree <- prediction(prob_dtree, labels = test_data$winner)
  dtree_auc_for_model <- performance(pred_dtree, measure='auc')
  dtree_auc_value_for_model <- dtree_auc_for_model@y.values
  return (dtree_auc_value_for_model)
}


################################################################################
################################      C5.0       ###############################
################################################################################

# Training the data on the C5.0 decision tree
ufc_dtree <- C5.0(training_data_without_class, training_data_only_class)

# Plotting decision tree
plot(ufc_dtree)

# Checking attribute usage
summary(ufc_dtree)

# Creating predictions
ufc_predictions_dtree <- predict(object = ufc_dtree, test_data)

# Evaluating performance
dtree_metrics  <- get_metrics(ufc_predictions_dtree)

# Getting AUC and performance model needed for plotting the ROC curve
auc_and_perf_dtree <- get_auc_and_perf(ufc_dtree)

dtree_metrics <- unlist(c(dtree_metrics, "AUC" = auc_and_perf_dtree[[1]]))

# IMPROVING PERFORMANCE

# Creating C5.0 decision trees with specified trials and 
# returning the predictions
dtree_creation <- function(trials){
  
  ufc_dtree_model <- C5.0(training_data_without_class, training_data_only_class, 
                          trials = trials)
  
  ufc_predictions_for_model <- predict(ufc_dtree_model, test_data)
  
  return (list(ufc_dtree_model, ufc_predictions_for_model))
}

# Getting AUC for given decision tree
# creating 20 C5.0 decision trees
dtree_models_and_predictions <- lapply(seq(1,20), dtree_creation) 

# Separating models and predictions
dtree_models <- lapply(dtree_models_and_predictions, `[[`, 1) 
dtree_predictions <- lapply(dtree_models_and_predictions, `[[`, 2)

# Getting metrics (accuracy, sensitivity and specificity)
metrics_for_dtrees <- lapply(dtree_predictions, get_metrics)

# Getting AUC
auc_for_dtrees <- lapply(dtree_models, get_auc)

# Converting to dataframes
metrics_for_dtree_per_trial <- as.data.frame(do.call(rbind, metrics_for_dtrees))

# creating dataframe and transposing it
auc_for_dtrees_df <- t(as.data.frame(auc_for_dtrees)) 

# changing column name
colnames(auc_for_dtrees_df) <- 'AUC'

# changing row names
rownames(auc_for_dtrees_df) <- 1:20 

#Merging two dataframes
all_metrics_dtree <- cbind(metrics_for_dtree_per_trial, auc_for_dtrees_df)

# getting the number of trials that produces the maximum AUC
highest_auc_index_dtree <- which(all_metrics_dtree$AUC==max(all_metrics_dtree[,4]))

#Creating model with the trials that had maximum AUC
ufc_dtree_final <- C5.0(training_data_without_class, 
                        training_data_only_class, 
                        trials=highest_auc_index_dtree)

# Evaluating performance
plot(ufc_dtree_final)

# Checking attribute usage
summary(ufc_dtree_final)

# Creating predictions
ufc_predictions_dtree_final <- predict(ufc_dtree_final, test_data)

# Evaluating performance
dtree_final_metrics  <- get_metrics(ufc_predictions_dtree_final)
# Getting AUC and performance model needed for plotting the ROC curve
auc_and_perf_dtree_final <- get_auc_and_perf(ufc_dtree_final)
# Setting accuracy, sensitivity, specificity and AUC in one variable
dtree_final_metrics <- unlist(c(dtree_final_metrics, 
                                "AUC" = auc_and_perf_dtree_final[[1]]))

################################################################################
#############################    RANDOM FOREST    ##############################
################################################################################

# creating RF model
ufc_rf <- randomForest(training_data_without_class, training_data_only_class,
                       ntree = 500,
                       mtry = sqrt(length(training_data_without_class)),
                       importance=TRUE)

# creating predictions
ufc_predictions_rf <- predict(ufc_rf, test_data)

# Evaluating performance
# Getting accuracy, sensitivity and specificity
rf_metrics <- get_metrics(ufc_predictions_rf)
# Getting AUC and performance object that is needed for plotting the ROC curve
auc_and_perf_rf <- get_auc_and_perf(ufc_rf)
# Storing all 4 metrics in one variable
rf_metrics <- unlist(c(rf_metrics, "AUC" = auc_and_perf_rf[[1]]))

# IMPROVING PERFORMANCE

# Creating random forests with specified number of trees and number of features
# returning those models with their predictions

rf_creation <- function(number_of_trees, number_of_features){
  
  ufc_rf_model <- randomForest(training_data_without_class, 
                               training_data_only_class, 
                               ntree = number_of_trees, 
                               mtry = number_of_features, 
                               importance=TRUE)
  
  ufc_predictions_for_model <- predict(ufc_rf_model, test_data)
  
  return (list(ufc_rf_model, ufc_predictions_for_model))
}

# Setting possible values for number of trees for each tree
number_of_trees <- seq(100,1000,100)
# Setting possible values for number of features for each tree
number_of_features <- c(sqrt(length(training_data_without_class)), 10, 15, 20)

# All possible combinations of given number of trees and features
combinations <- expand.grid(number_of_trees, number_of_features) 
# Separating the number of trees and number of features as separate lists
n_of_trees <- combinations$Var1
n_of_features <- combinations$Var2

#creating 10*4 = 40 random forests
rf_models_and_predictions <- lapply(n_of_trees, rf_creation, n_of_features)

#Separating models and predictions
rf_models <- lapply(rf_models_and_predictions, `[[`, 1) 
rf_predictions <- lapply(rf_models_and_predictions, `[[`, 2)

#Getting metrics (accuracy, sensitivity and specificity)
metrics_for_rf <- lapply(rf_predictions, get_metrics) 

#Getting AUC
auc_for_rf <- lapply(rf_models, get_auc)

#Converting to dataframes
metrics_for_rf_per_trial <- as.data.frame(do.call(rbind, metrics_for_rf))
#creating dataframe and transposing it
auc_for_rf_df <- t(as.data.frame(auc_for_rf)) 
colnames(auc_for_rf_df) <- 'AUC' #changing column name
rownames(auc_for_rf_df) <- 1:40 #changing row names

#Merging two dataframes
all_metrics_rf <- cbind(metrics_for_rf_per_trial, auc_for_rf_df)

# getting index of all_metrics_rf with max AUC
highest_auc_index_rf <- which(all_metrics_rf$AUC==max(all_metrics_rf[,4]))

# getting the number of trees and number of features combination that produces the maximum AUC
best_number_of_trees <- combinations[highest_auc_index_rf,]$Var1
best_number_of_features <- combinations[highest_auc_index_rf,]$Var2

#Creating random forest with these hyperparameters
ufc_rf_final <- randomForest(training_data_without_class, 
                             training_data_only_class, 
                             ntree = best_number_of_trees, mtry = best_number_of_features)

# Creating predictions
ufc_rf_final_predictions <- predict(ufc_rf_final, test_data)

# Evaluating performance
rf_final_metrics <- get_metrics(ufc_rf_final_predictions)

# geting auc value and performance object for ploting ROC curve
auc_and_perf_rf_final <- get_auc_and_perf(ufc_rf_final)
# Storing all 4 metrics in one variable
rf_final_metrics <- unlist(c(rf_final_metrics,
                             "AUC" = auc_and_perf_rf_final[[1]]))

# Checking the importance of features for random forest
varImp(ufc_rf_final)
# Feature selection will be done at the end

################################################################################
####################################   KNN   ###################################
################################################################################

#min_max normalization function
min_max_norm <- function(x){
  ((x-min(x))/(max(x)-min(x)))
}

#normalizing the data
df_normalized <- as.data.frame(lapply(df[,-1], min_max_norm))

# Splitting on training and test set
training_data_knn <- df_normalized[sample_, ] 
test_data_knn <- df_normalized[-sample_, ] 

# training the data on KNN
# rule of thumb for choosing K is sqrt of data length
ufc_knn <- knn(train = training_data_knn, 
               test = test_data_knn, 
               cl = training_data_only_class, 
               k = round(sqrt(length(training_data$winner))), 
               prob=TRUE)

# Evaluating performance
knn_metrics <- get_metrics(ufc_knn)

## Getting AUC value for and plotting the ROC Curve for KNN (from class library)
prob_knn <- attr(ufc_knn, 'prob')
prob_knn <- 2*ifelse(ufc_knn == '-1', 1-prob_knn, prob_knn) - 1
pred_knn <- prediction(prob_knn, test_data$winner)
performance_knn <- performance(pred_knn, 'tpr','fpr')

#Getting the AUC value for KNN
knn_auc <- performance(pred_knn, measure='auc')

knn_auc_value <- knn_auc@y.values 
# Storing all 4 metrics in one variable
knn_metrics <- unlist(c(knn_metrics, 'AUC' = knn_auc_value))

################################################################################
############################   TRAIN CONTROL OBJECTS    ########################
################################################################################

# creating set of configuration options for train() function
train_control_cv <- trainControl(method='cv',
                                 number = 10,
                                 classProbs = TRUE,
                                 summaryFunction = mix_summary,
                                 savePredictions = TRUE)
# Train control with repeated CV
train_control_repeated_cv <- trainControl(method='repeatedcv',
                                          number = 3,
                                          repeats = 4,
                                          classProbs = TRUE,
                                          summaryFunction = mix_summary,
                                          savePredictions = TRUE)

train_control_bootstrap <- trainControl(method='boot',
                                        number = 20,
                                        classProbs = TRUE,
                                        summaryFunction = mix_summary,
                                        savePredictions = TRUE)

# Will be used for the final model with optimized hyperparameters
train_control_LOOCV <- trainControl(method='LOOCV',
                                    classProbs = TRUE,
                                    summaryFunction = mix_summary,
                                    savePredictions = TRUE)

################################################################################
######################   C5.0 WITH DIFFERENT RESAMPLING METHODS   ##############
################################################################################
# Creates C50 decision tree model
# Returns 4 metrics and performance object that is required for plotting the ROC curve
train_c50 <- function(train_control_obj) {
  
  # creating C5.0 model 
  model <- train (training_data_without_class,
                  training_data_only_class,
                  data = training_data, 
                  method ='C5.0', 
                  trControl = train_control_obj, 
                  metric = 'ROC')
  
  prediction <- predict(model, test_data)
  
  auc_and_performance <-get_auc_and_perf(model)
  
  metrics <- c(get_metrics(prediction), 'AUC' = auc_and_performance[[1]])
  
  return(list(metrics, auc_and_performance[[2]]))
}

# Using cross validation
ufc_dtree_cv_10 <- train_c50(train_control_cv)

# Evaluating performance
dtree_cv_metrics <- ufc_dtree_cv_10[[1]]

# Using repeated cross validation
ufc_dtree_repeated_cv <- train_c50(train_control_repeated_cv)

# Evaluating performance
dtree_repeated_cv_metrics <- ufc_dtree_repeated_cv[[1]]

# Using bootstrap
ufc_dtree_boot <- train_c50(train_control_bootstrap)

# Evaluating performance
dtree_boot_metrics <- ufc_dtree_boot[[1]]


################################################################################
############    RANDOM FOREST WITH DIFFERENT RESAMPLING METHODS   ##############
################################################################################

train_rf <- function(train_control_obj) {
  model <- train (training_data_without_class,
                  training_data_only_class,
                  data = training_data, 
                  method ='rf', 
                  tuneGrid= expand.grid(.mtry=10), 
                  trControl = train_control_obj, 
                  metric = 'ROC')
  
  prediction <- predict(model, test_data)
  
  auc_and_performance <-get_auc_and_perf(model)
  
  metrics <- c(get_metrics(prediction), 'AUC' = auc_and_performance[[1]])
  
  return(list(metrics, auc_and_performance[[2]]))
}

# Using cross validation
ufc_rf_cv_10 <- train_rf(train_control_cv)

# Evaluating performance
rf_cv_metrics <- ufc_rf_cv_10[[1]]

# Using repeated cross validation
ufc_rf_repeated_cv <- train_rf(train_control_repeated_cv)

# Evaluating performance
rf_repeated_cv_metrics <- ufc_rf_repeated_cv[[1]]

# Using bootstrap
ufc_rf_boot <- train_rf(train_control_bootstrap)

# Evaluating performance
rf_boot_metrics <- ufc_rf_boot[[1]]

################################################################################
##############################   10 FOLD CV KNN   ##############################
################################################################################

# Function for creating 10-fold CV KNN
# Returns metrics and performance object that is used for plotting ROC curve
train_knn <- function(train_control_obj) {
  
  model <-  train (winner ~ .,
                   data = training_data, 
                   method ="knn", 
                   trControl = train_control_obj, 
                   preProcess = c("center", "scale"),
                   tuneGrid = expand.grid(k = seq(3, 47, 2)),
                   metric = 'ROC')
  
  prediction <- predict(model, test_data)
  
  auc_and_performance <-get_auc_and_perf(model)
  
  metrics <- c(get_metrics(prediction), 'AUC' = auc_and_performance[[1]])
  
  return(list(metrics, performance_knn))
}

# Using cross validation
ufc_knn_cv_10 <- train_knn(train_control_cv)

# Evaluating performance
knn_cv_metrics <- ufc_knn_cv_10[[1]]

plot(ufc_knn_cv_10[[2]])

# Using repeated cross validation
ufc_knn_repeated_cv <- train_knn(train_control_repeated_cv)

# Evaluating performance
knn_repeated_cv_metrics <- ufc_knn_repeated_cv[[1]]

# Using bootstrap
ufc_knn_boot <- train_knn(train_control_bootstrap)

# Evaluating performance
knn_boot_metrics <- ufc_knn_boot[[1]]

################################################################################
###############################  ALL ROC CURVES  ############################### 
################################################################################

# Plotting the ROC curve for C5.0 with default hyperparameter
plot(auc_and_perf_dtree[[2]], 
     main='ROC Curve for C5.0 with default trials hyperparameter')

# Plotting the ROC curve with the trials that had maximum AUC
plot(auc_and_perf_dtree_final[[2]], 
     main=paste('ROC Curve for Decision tree with trials = ', 
                highest_auc_index_dtree))

# Plotting the ROC curve for RF with default hyperparameters
plot(auc_and_perf_rf[[2]], main='ROC Curve for RF with default parameters')

# Plotting the ROC curve with optimal hyperparameters
plot(auc_and_perf_rf_final[[2]], 
     main=paste('ROC Curve for RF with ntree = ', best_number_of_trees, 
                ' and mtry = ', best_number_of_features))

# Plotting the ROC Curve for KNN
plot(performance_knn, avg='threshold',
     main=paste('ROC Curve for KNN with k = ',
                round(sqrt(length(training_data$winner)))))

# Plotting the ROC curves for C5.0

# Plotting the ROC curve for 10-fold CV C5.0 model
plot(ufc_dtree_cv_10[[2]], main = 'C5.0 with 10-fold CV')
# Plotting the ROC curve for repeated CV C5.0 model
plot(ufc_dtree_repeated_cv[[2]], main = 'C5.0 with repeated CV')
# Plotting the ROC curve for C5.0 model with bootstrap sampling
plot(ufc_dtree_boot[[2]], main = 'C5.0 bootstrap sampling')

# Plotting the ROC curves for RF

# Plotting the ROC curve for 10-fold CV RF model
plot(ufc_rf_cv_10[[2]], main = 'Random Forest with 10-fold CV')
# Plotting the ROC curve for repeated CV RF model
plot(ufc_rf_repeated_cv[[2]], main = 'Random Forest repeated CV')
# Plotting the ROC curve for RF model with bootstrap sampling
plot(ufc_rf_boot[[2]], main = 'Random Forest with bootstrap sampling')

# Plotting the ROC curves for KNN

# Plotting the ROC curve for 10-fold CV RF model
plot(ufc_knn_cv_10[[2]], main = 'KNN with 10-fold CV')
# Plotting the ROC curve for repeated CV RF model
plot(ufc_knn_repeated_cv[[2]], main = 'KNN repeated CV')
# Plotting the ROC curve for RF model with bootstrap sampling
plot(ufc_knn_boot[[2]], main = 'KNN with bootstrap sampling')

# Plotting all ROC curves together on one plot
x_values_all <- c(auc_and_perf_dtree[[2]]@x.values[[1]], 
                  auc_and_perf_dtree_final[[2]]@x.values[[1]],
                  auc_and_perf_rf[[2]]@x.values[[1]],
                  auc_and_perf_rf_final[[2]]@x.values[[1]],
                  performance_knn@x.values[[1]],
                  ufc_dtree_cv_10[[2]]@x.values[[1]],
                  ufc_dtree_repeated_cv[[2]]@x.values[[1]],
                  ufc_dtree_boot[[2]]@x.values[[1]],
                  ufc_rf_cv_10[[2]]@x.values[[1]],
                  ufc_rf_repeated_cv[[2]]@x.values[[1]],
                  ufc_rf_boot[[2]]@x.values[[1]],
                  ufc_knn_cv_10[[2]]@x.values[[1]],
                  ufc_knn_repeated_cv[[2]]@x.values[[1]],
                  ufc_knn_boot[[2]]@x.values[[1]])

y_values_all <- c(auc_and_perf_dtree[[2]]@y.values[[1]], 
                  auc_and_perf_dtree_final[[2]]@y.values[[1]],
                  auc_and_perf_rf[[2]]@y.values[[1]],
                  auc_and_perf_rf_final[[2]]@y.values[[1]],
                  performance_knn@y.values[[1]],
                  ufc_dtree_cv_10[[2]]@y.values[[1]],
                  ufc_dtree_repeated_cv[[2]]@y.values[[1]],
                  ufc_dtree_boot[[2]]@y.values[[1]],
                  ufc_rf_cv_10[[2]]@y.values[[1]],
                  ufc_rf_repeated_cv[[2]]@y.values[[1]],
                  ufc_rf_boot[[2]]@y.values[[1]],
                  ufc_knn_cv_10[[2]]@y.values[[1]],
                  ufc_knn_repeated_cv[[2]]@y.values[[1]],
                  ufc_knn_boot[[2]]@y.values[[1]])

id_column <- c(rep('Decision Tree (Default)', 
                   length(auc_and_perf_dtree[[2]]@x.values[[1]])),
               rep('Decision Tree (Final)', 
                   length(
                     auc_and_perf_dtree_final[[2]]@x.values[[1]])),
               rep('Random Forest (Default)', 
                   length(auc_and_perf_rf[[2]]@x.values[[1]])),
               rep('Random Forest (Final)', 
                   length(
                     auc_and_perf_rf_final[[2]]@x.values[[1]])),
               rep('KNN (Default)', 
                   length(performance_knn@x.values[[1]])),
               rep('Decision Tree (CV)', 
                   length(ufc_dtree_cv_10[[2]]@x.values[[1]])),
               rep('Decision Tree (Repeated CV)', 
                   length(
                     ufc_dtree_repeated_cv[[2]]@x.values[[1]])),
               rep('Decision Tree (Bootstrap)', 
                   length(ufc_dtree_boot[[2]]@x.values[[1]])),
               rep('Random Forest (CV)', 
                   length(ufc_rf_cv_10[[2]]@x.values[[1]])),
               rep('Random Forest (Repeated CV)', 
                   length(ufc_rf_repeated_cv[[2]]@x.values[[1]])),
               rep('Random Forest (Bootstrap)', 
                   length(ufc_rf_boot[[2]]@x.values[[1]])),
               rep('KNN (CV)', 
                   length(ufc_knn_cv_10[[2]]@x.values[[1]])),
               rep('KNN (Repeated CV)', 
                   length(ufc_knn_repeated_cv[[2]]@x.values[[1]])),
               rep('KNN (Bootstrap)', 
                   length(ufc_knn_boot[[2]]@x.values[[1]])))

roc_df <- data.frame(x=x_values_all, y=y_values_all, Model=id_column)

# Plotting the actual ggplot
ggplot(roc_df, aes(x = x, y = y, color = Model)) +
  geom_line(size = 1) +
  geom_abline(intercept = 0, slope = 1) +
  scale_color_manual(values = c('#e6194b', '#3cb44b', '#ffe119', '#4363d8', 
                                '#f58231', '#911eb4', '#46f0f0', '#f032e6', 
                                '#bcf60c', '#fabebe', '#008080', '#808080', 
                                '#910606', '#000000')) +
  labs(title = "All ROC curves plotted", 
       x = "False Positive Rate", 
       y = "True Positive Rate") +
  theme_light() +
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 20),
        legend.position = c(0.87, 0.3))

##############################     ALL METRICS    ##############################

all_metrics <- data.frame(dtree_metrics, dtree_final_metrics, dtree_cv_metrics,
                          dtree_repeated_cv_metrics, dtree_boot_metrics,
                          rf_metrics, rf_final_metrics, rf_cv_metrics, 
                          rf_repeated_cv_metrics, rf_boot_metrics, 
                          knn_metrics, knn_cv_metrics, knn_repeated_cv_metrics,
                          knn_boot_metrics)

# Export metrics to the csv to use it in the presentation/report.
all_metrics_t <- t(as.data.frame(all_metrics))
write.csv(all_metrics_t, file = "final_metrics_rf.csv", sep = ",")

