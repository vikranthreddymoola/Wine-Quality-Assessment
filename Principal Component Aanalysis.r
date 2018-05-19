set.seed(10)
library(caret)
library(kknn)
library(corrplot)
library(randomForest)
library(kernlab)

#importing the red-wine quality dataset from the UCI Website
red_wine_url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
red_wine_rawdata <- read.csv(red_wine_url, header = TRUE, sep = ";")
red_wine <- red_wine_rawdata
str(red_wine)
table(red_wine$quality)

#MODEL BUILDING
#Splitting the data in to training and test sets
red_wine$quality <- as.numeric(red_wine$quality)
training <- sample(1:nrow(red_wine), 1066)
pca_training_red_wine <- red_wine[training, ]
pca_testing_red_wine <- red_wine[-training, ]

#PCA
principle_comp <- prcomp(pca_training_red_wine, scale. = T)
summary(principle_comp)

#Dimensions of principal component score vectors
dim(principle_comp$x)

#resultant principle component parts
biplot(principle_comp, scale = 0)

#standard deviation and variance of each principle component
standard_deviation <- principle_comp$sdev
principle_variance <- standard_deviation^2

#Proportion of Variance explained
proportion_varex <- principle_variance/sum(principle_variance)
proportion_varex

#Screen Plot
plot(proportion_varex, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")

#Cumulative screen plot
plot(cumsum(proportion_varex), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

