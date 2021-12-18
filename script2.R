rm(list=ls())

# Importing necessarz libraries
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
  return (output)
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

auc_and_perf_dtree <- get_auc_and_perf(ufc_dtree)
dtree_metrics <- unlist(c(dtree_metrics, "AUC" = auc_and_perf_dtree[[1]]))

plot(auc_and_perf_dtree[[2]], main='ROC Curve for Dtree')


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
# with trials=20 we have the overall best result 
# (best AUC, accuracy and sensitivity )
max(all_metrics_dtree[,4]) #AUC = 0.8973

#Creating model with trials = 20
ufc_dtree_20 <- C5.0(training_data_without_class, 
                     training_data_only_class, 
                     trials=20)

#evaluating performance
plot(ufc_dtree_20)

# Checking attribute usage
summary(ufc_dtree_20)

# Creating predictions
ufc_predictions_dtree_20 <- predict(ufc_dtree_20, test_data)

# Evaluating performance
dtree_20_metrics  <- get_metrics(ufc_predictions_dtree_20)

auc_and_perf_dtree_20 <- get_auc_and_perf(ufc_dtree_20)

dtree_20_metrics <- unlist(c(dtree_20_metrics, 
                             "AUC" = auc_and_perf_dtree_20[[1]]))

plot(auc_and_perf_dtree_20[[2]], 
     main='ROC Curve for Decision tree with trials = 20')


################################################################################
#############################    RANDOM FOREST    ##############################
################################################################################

# creating model
ufc_rf <- randomForest(training_data_without_class, training_data_only_class,
                       ntree = 500,
                       mtry = sqrt(length(training_data_without_class)),
                       importance=TRUE)

# creating predictions
ufc_predictions_rf <- predict(ufc_rf, test_data)

# Evaluating performance
rf_metrics <- get_metrics(ufc_predictions_rf)

auc_and_perf_rf <- get_auc_and_perf(ufc_rf)
rf_metrics <- unlist(c(rf_metrics, "AUC" = auc_and_perf_rf[[1]]))

plot(auc_and_perf_rf[[2]], main='ROC Curve for RF')


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


number_of_trees <- seq(100,1000,100)

number_of_features <- c(sqrt(length(training_data_without_class)), 10, 15, 20)

#all possible combinations of given number of trees and features
combinations <- expand.grid(number_of_trees, number_of_features) 
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

# we picked the one with highest AUC = index 17 
# ntree = 700, mtry = 10
max(all_metrics_rf[,4]) # AUC = 0.926

#Creating random forest with these hyperparameters
ufc_rf_final <- randomForest(training_data_without_class, 
                             training_data_only_class, 
                             ntree = 700, mtry = 10)

# Creating predictions
ufc_rf_final_predictions <- predict(ufc_rf_final, test_data)

# Evaluating performance
rf_final_metrics <- get_metrics(ufc_rf_final_predictions)

# geting auc value and performance object for ploting ROC curve
auc_and_perf_rf_final <- get_auc_and_perf(ufc_rf_final)
rf_final_metrics <- unlist(c(rf_final_metrics,
                             "AUC" = auc_and_perf_rf_final[[1]]))

plot(auc_and_perf_rf_final[[2]], main='ROC Curve for RF')

################################################################################
####################################   KNN   ###################################
################################################################################

#min_max normalization function
min_max_norm <- function(x){
  ((x-min(x))/(max(x)-min(x)))
}

#normalizing the data
df_normalized <- as.data.frame(lapply(df[,-1], min_max_norm))
training_data_knn <- df_normalized[sample_, ] 
test_data_knn <- df_normalized[-sample_, ] 

# training the data on KNN
# rule of thumb for choosing K is sqrt of data length
ufc_knn <- knn(train = training_data_knn, 
               test = test_data_knn, 
               cl = training_data_only_class, 
               k = sqrt(length(training_data$winner)), 
               prob=TRUE)


# Evaluating performance
knn_metrics <- get_metrics(ufc_knn)

## Getting AUC value for and plotting the ROC Curve for KNN (from class library)
prob_knn <- attr(ufc_knn, 'prob')
prob_knn <- 2*ifelse(ufc_knn == '-1', 1-prob_knn, prob_knn) - 1
pred_knn <- prediction(prob_knn, test_data$winner)
performance_knn <- performance(pred_knn, 'tpr','fpr')

#Plotting the ROC Curve for KNN
plot(performance_knn, avg='threshold', lwd=3, main='ROC Curve for KNN')

#Getting the AUC value for KNN
knn_auc <- performance(pred_knn, measure='auc')
str(knn_auc) #0.6926078 AUC for KNN
knn_auc_value <- knn_auc@y.values 

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

train_control_repeated_cv <- trainControl(method='repeatedcv',
                                          number = 3,
                                          repeats = 4,
                                          classProbs = TRUE,
                                          summaryFunction = mix_summary,
                                          savePredictions = TRUE)
train_control_bootstrap <- trainControl(method='boot',
                                        number = 15,
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

train_c50 <- function(train_control_obj) {
  
  # creating C5.0 model 
  model <- train (training_data_without_class,
                  training_data_only_class,
                  data = training_data, 
                  method ='C5.0', 
                  trControl = train_control_obj, 
                  metric = 'ROC')
  
  prediction <- predict(model, test_data)
  
  evalm_obj <- evalm(model,gnames=c('model'), 
                     silent = TRUE, 
                     showplots = FALSE)
  
  metrics <- c(get_metrics(prediction),
               "AUC" = evalm_obj$stdres$`model`$Score[13])
  
  return(list(model, metrics))
}

# Using cross validation
ufc_dtree_cv_10 <- train_c50(train_control_cv)

# Evaluating performance
dtree_cv_metrics <- ufc_dtree_cv_10[[2]]

# Using repeated cross validation
ufc_dtree_repeated_cv <- train_c50(train_control_repeated_cv)

# Evaluating performance
dtree_repeated_cv_metrics <- ufc_dtree_repeated_cv[[2]]

# Using bootstrap
ufc_dtree_boot <- train_c50(train_control_bootstrap)

# Evaluating performance
dtree_boot_metrics <- ufc_dtree_boot[[2]]

################################################################################
#########################   10-FOLD CV RANDOM FOREST   #########################
################################################################################

# creating RF model with 10-fold cross validation
ufc_rf_cv_10 <- train (training_data_without_class,
                       training_data_only_class,
                       data = training_data, 
                       method ='rf', 
                       trControl = train_control_cv, 
                       metric = 'ROC')

# creating predictions  
pred_rf_cv <- predict(ufc_rf_cv_10, test_data)

# Evaluating performance
rf_cv_metrics <- get_metrics(pred_rf_cv)

################################################################################
##############################   10 FOLD CV KNN   ##############################
################################################################################

ufc_knn_cv_10 <- train (winner ~ .,
                        data = training_data, 
                        method ="knn", 
                        trControl = train_control_cv, 
                        preProcess = c("center", "scale"),
                        tuneGrid = expand.grid(k = 1:50),
                        metric = 'ROC')
plot(ufc_knn_cv_10)

# creating predictions
pred_knn_cv <- predict(ufc_knn_cv_10, test_data)

# evaluating preformance
knn_cv_metrics <- get_metrics(pred_knn_cv)



################################################################################
############## ROC Curves for models with cross validation  ####################
################################################################################

roc_curves <- evalm(list(ufc_dtree_cv_10, ufc_rf_cv_10, ufc_knn_cv_10),
                    gnames=c('DTree CV10','RF CV10', 'KNN CV10'))
# AUC for DTree CV 10 = 0.9 roc_curves$stdres$`DTree CV10`$Score[13]
# AUC for RF CV 10 = 0.91 roc_curves$stdres$`RF CV10`$Score[13]
# AUC for KNN CV 10 = 0.88 roc_curves$stdres$`DTree CV10`$Score[13]
#roc curve
roc_curves$roc

dtree_cv_metrics <-  unlist(c(dtree_cv_metrics,
                              "AUC" = roc_curves$stdres$`DTree CV10`$Score[13]))
rf_cv_metrics <-  unlist(c(rf_cv_metrics, 
                           "AUC" = roc_curves$stdres$`RF CV10`$Score[13]))
knn_cv_metrics <-  unlist(c(knn_cv_metrics, 
                            "AUC" = roc_curves$stdres$`DTree CV10`$Score[13]))


##############################     ALL METRICS    ##############################

all_metrics <- data.frame(dtree_metrics, rf_metrics, dtree_cv_metrics,
                          rf_cv_metrics, knn_metrics, knn_cv_metrics) 
                          #rf_final_metrics, dtree_20_metrics)


# Checking the importane of features for random forest
varImp(ufc_rf_final)
# threshold 10
# Red_sig_str_rnd_1 and blue
# red_tot_str_rnd_1 and blue
# red_tot_str_attempt_rnd_1 and blue
# red_ctrl_time_rnd_1 and blue
# red_sig_str_rnd_2                 
# red_tot_str_rnd_2                 
# red_tot_str_attempt_rnd_2
# red_ctrl_time_rnd_2               
# red_sig_str_rnd_3                 
# red_tot_str_rnd_3
# red_tot_str_attempt_rnd_3
# red_tot_str_attempt_rnd_3 (blue has 9.84, but we will include it because red has 13.59)
# red_ctrl_time_rnd_3
# red_sig_str_head_rnd_1
# red_sig_str_head_rnd_2
# red_sig_str_head_rnd_3    
# 16*2 = 32 features?

