df <- read.csv("ufcstats_cleaned.csv", header = TRUE, sep = ",")

str(df) 
set.seed(1)

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

## Evaluating performance

# Creating predictions
ufc_predictions <- predict(object = ufc_dtree, test_data)
ufc_predictions
test_data$winner


library(gmodels)
#confusion matrix #1, absence is "positive" class
CrossTable(test_data$winner, ufc_predictions, 
           prop.chisq = FALSE, dnn=c('Actual', 'Predicted'))
library(caret)
#confusion matrix # positive is 1 (absence)
confMatrixObject <- confusionMatrix(as.factor(ufc_predictions), 
                                    as.factor(test_data$winner), 
                                    positive = "blue", 
                                    dnn= c("Predicted", "Reference/Actual"))
confMatrixObject$overall['Accuracy']
confMatrixObject$byClass['Sensitivity']
confMatrixObject$byClass['Specificity']





