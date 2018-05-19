set.seed(10)
library(caret)
library(kknn)
library(corrplot)
library(randomForest)
library(kernlab)

#importing the red-wine quality dataset from the UCI Machine Learning Repository
red_wine_url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
red_wine_rawdata <- read.csv(red_wine_url, header = TRUE, sep = ";")
red_wine <- red_wine_rawdata
str(red_wine)
table(red_wine$quality)

#plots for each predictor variable
red_wine$quality <- as.integer(red_wine$quality)
par(mfrow = c(4,3))
j=1
while (j<12){ 
  plot(red_wine[, j], jitter(red_wine[, "quality"]), xlab = names(red_wine)[j],
       ylab = "quality", col = "violet", cex = 0.8, cex.lab = 1.3)
  abline(lm(red_wine[, "quality"] ~ red_wine[ ,j]), lty = 2, lwd = 2)
  j=j+1
}
par(mfrow = c(1, 1))

#plot for the correlation of each variable
par(mfrow = c(1,1))
correlation_red_wine <- cor(red_wine)
corrplot(correlation_red_wine, method = 'square')

#MODEL BUILDING
#Splitting the data in to training and test sets
red_wine$quality <- as.factor(red_wine$quality)
training <- sample(1:nrow(red_wine), 1066)
training_red_wine <- red_wine[training, ]
testing_red_wine <- red_wine[-training, ]

#SVM
train_ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 5)
svm_grid <- expand.grid(C = c(1:10), sigma = seq(0.1, 1, length = 10))
svm_train <- train(quality ~ ., data = training_red_wine, method = "svmRadial",
                   trControl = train_ctrl, tuneGrid = svm_grid,
                   preProcess = c("center", "scale"))
plot(svm_train)
svm_train$bestTune
svm_predict <- predict(svm_train, testing_red_wine)
confusionMatrix(svm_predict, testing_red_wine$quality)

#knn
train_ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 5)
knn_grid <- expand.grid(kmax = c(3, 5, 7, 9, 11), distance = c(1, 2),
                         kernel = c("rectangular", "cos", "gaussian"))
knn_train <- train(quality ~ ., data = training_red_wine, method = "kknn",
                    trControl = train_ctrl, tuneGrid = knn_grid,
                    preProcess = c("center", "scale"))
plot(knn_train)
knn_train$bestTune
knn_predict <- predict(knn_train, testing_red_wine)
confusionMatrix(knn_predict, testing_red_wine$quality)

#randomForest
train_ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 5)
rf_grid <- expand.grid(mtry = 1:11)
rf_train <- train(quality ~ ., data = training_red_wine, method = "rf",
                  trControl = train_ctrl, tuneGrid = rf_grid, 
                  preProcess = c("center", "scale"))
plot(rf_train)
rf_train$bestTune
rf_predict <- predict(rf_train, testing_red_wine)
confusionMatrix(rf_predict, testing_red_wine$quality)
 #feature importance 
importance <- varImp(rf_train, scale=FALSE)
print(importance)
plot(importance, ylab="Atributes")

#Neural Networks
train_ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 5)
nnet_train<-train(quality~.,data=training_red_wine,method="nnet",trControl=train_ctrl,tuneGrid=NULL,preProcess=c("center","scale"))
plot(nnet_train)
nnet_train$bestTune
nnet_predict <- predict(nnet_train, testing_red_wine)
confusionMatrix(nnet_predict, testing_red_wine$quality)



