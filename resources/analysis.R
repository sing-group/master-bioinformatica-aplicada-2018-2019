#################################################
#	Data loading
################################################# 

setwd(Sys.getenv("WORK_DIR"))
data <- read.csv("data/wdbc.data")

data$diagnosis <- factor(data$diagnosis)
data <- data[,c(2:ncol(data))]

dim(data)
head(data)
summary(data)

set.seed(2019)

trainSamples <- round(0.7*nrow(data))
trainSamplesIndexes <- sample(1:nrow(data), trainSamples)

train <- data[trainSamplesIndexes,]
test <- data[-trainSamplesIndexes,]

#################################################
#	Decision Tree
#################################################

library(rpart)
library(rpart.plot)

tree <- rpart(diagnosis~., method = 'class', data=train)
tree

# Simple plot: https://stat.ethz.ch/R-manual/R-devel/library/rpart/html/plot.rpart.html
png("tree-1.png")
plot(tree, uniform=TRUE, margin=.05)
text(tree, use.n = TRUE)
dev.off()

png("tree-2.png")
rpart.plot(tree, box.palette="RdBu", shadow.col="gray", nn=TRUE)
dev.off()

# Train data predictions
tree.pred.train <- predict(tree, type = 'class')
table(tree.pred.train, train$diagnosis)

# Test data predictions
tree.pred.test <- predict(tree, newdata=test, type = 'class')
table(tree.pred.test, test$diagnosis)

# Try plot some of the variables of the trained model
png("concave.points_worst.png", width = 1500, height = 500)
par(mfrow=c(1,3))
boxplot(concave.points_worst ~ diagnosis, data=data, col=(c("gold","darkgreen")), main = "All data")
boxplot(concave.points_worst ~ diagnosis, data=train, col=(c("gold","darkgreen")), main = "Train")
boxplot(concave.points_worst ~ diagnosis, data=test, col=(c("gold","darkgreen")), main = "Test")
dev.off()

png("area_worst.png", width = 1500, height = 500)
par(mfrow=c(1,3))
boxplot(area_worst ~ diagnosis, data=data, col=(c("gold","darkgreen")), main = "All data")
boxplot(area_worst ~ diagnosis, data=train, col=(c("gold","darkgreen")), main = "Train")
boxplot(area_worst ~ diagnosis, data=test, col=(c("gold","darkgreen")), main = "Test")
dev.off()

#################################################
#	Random Forest
#################################################
library(randomForest)
library(ggplot2)

randomforest <- randomForest(diagnosis~., data=train)
randomforest

# Train data predictions
randomforest.pred.train <- predict(randomforest)
table(randomforest.pred.train, train$diagnosis)

# Test data predictions
randomforest.pred.test <- predict(randomforest, newdata=test)
table(randomforest.pred.test, test$diagnosis)

# Train a model with different numbers of trees
for(ntree in c(500,1000,1500,2000)) {
	print(randomForest(diagnosis~., data=train, ntree=ntree))
}

# Out-of-bag (OOB) error
error_df <- data.frame(error_rate = randomforest$err.rate[,'OOB'], num_trees = 1:randomforest$ntree)
png("random-forest-oob-error.png")
ggplot(error_df, aes(x=num_trees, y=error_rate)) + geom_line()
dev.off()

#################################################
#	SVM
#################################################
library("e1071")
library("ggplot2")

# Train the model based only in two variables
svm.model <- svm(diagnosis ~ concave.points_worst + area_worst, data = train) 
svm.model

# Train data predictions
svm.pred.train <- predict(svm.model)
table(svm.pred.train, train$diagnosis)

# Test data predictions
svm.pred.test <- predict(svm.model, newdata=test)
table(svm.pred.test, test$diagnosis)

png("svm-data.png")
ggplot(data = train, aes(x = area_worst, y = concave.points_worst, color = diagnosis, shape = diagnosis)) + 
  geom_point(size = 2) +
  scale_color_manual(values=c("#000000", "#FF0000")) +
  theme(legend.position = "none")
dev.off()

png("svm-classifier.png")
plot(svm.model, data=train, concave.points_worst ~ area_worst)
dev.off()

svm.model.all <- svm(diagnosis~., data=train)
svm.model.all

# Train data predictions
svm.all.pred.train <- predict(svm.model.all)
table(svm.all.pred.train, train$diagnosis)

# Test data predictions
svm.all.pred.test <- predict(svm.model.all, newdata=test)
table(svm.all.pred.test, test$diagnosis)

#################################################
#	Naive Bayes
#################################################
library("e1071")

naivebayes <- naiveBayes(diagnosis~., data=train)
naivebayes

# Train data predictions
naivebayes.pred.train <- predict(naivebayes, train)
table(naivebayes.pred.train, train$diagnosis)

# Test data predictions
naivebayes.pred.test <- predict(naivebayes, newdata=test)
table(naivebayes.pred.test, test$diagnosis)

# Get raw predictions
naivebayes.pred.raw.test <- predict(naivebayes, newdata=test, type="raw")
naivebayes.pred.raw.test

# Compare raw predictions with predicted and actual classes
equals <- (naivebayes.pred.test == test$diagnosis)
comparisons <- data.frame(naivebayes.pred.raw.test, naivebayes.pred.test, test$diagnosis, equals)
comparisons

#################################################
#	Evaluating Classification Models
#################################################
pred <- naivebayes.pred.test

# Confusion matrix

pred_y <- as.numeric(pred == "M")
true_y <- as.numeric(test$diagnosis == "M")
true_positive <- (true_y==1) & (pred_y==1)
true_negative <- (true_y==0) & (pred_y==0)
false_positive <- (true_y==0) & (pred_y==1)
false_negative <- (true_y==1) & (pred_y==0)

confusion_matrix <- matrix(c(
		sum(true_negative), sum(false_positive),
		sum(false_negative), sum(true_positive)
	),2,2
)
rownames(confusion_matrix) <- c("Predicted: B", "Predicted: M")
colnames(confusion_matrix) <- c("Actual: B", "Actual: M")

# Compare this confusion matrix with the one created using the table function
confusion_matrix
table(pred, test$diagnosis)

# Precision, Recall (Sensitivity) and Specificity

# Precision is the percentage of samples predicted as positive ("M") that are 
# actually positive.
(precision <- sum(true_positive) / (sum(true_positive) + sum(false_positive)))

# Recall (or Sensitivity) is the proportion of positive samples ("M") that are
# identified correctly (True Positive Rate)
(recall <- sum(true_positive) / (sum(true_positive) + sum(false_negative)))

# Specificity is the proportion of negative samples ("B") that are
# identified correctly (True Negative Rate)
(specificity <- sum(true_negative) / (sum(true_negative) + sum(false_positive)))

# ROC Curves

library(ggplot2)

pred <- naivebayes.pred.raw.test
pred <- pred[,2]

idx <- order(-pred)
recall <- cumsum(true_y[idx]==1)/sum(true_y==1)
specificity <- (sum(true_y==0) - cumsum(true_y[idx]==0)) / sum(true_y==0)
roc_df <- data.frame(recall = recall, specificity = specificity)

png("roc-1.png")
plot(roc_df$specificity, roc_df$recall, xlim = rev(range(roc_df$specificity)), type="l")
dev.off()

png("roc-2.png")
ggplot(roc_df, aes(x=specificity, y=recall)) +
	geom_line(color='blue') +
	scale_x_reverse(expand=c(0, 0)) +
	scale_y_continuous(expand=c(0, 0)) +
	geom_line(data = data.frame(x=(0:100)/100), aes(x=x, y=1-x), linetype='dotted', color='red')
dev.off()
