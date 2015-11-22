---
title: "Practical Machine Learning Project"
author: "jemin lee (Ph.D Candidate, Chungnam National University, South Korea)"
date: "2015년 11월 22일"
output: html_document
---

Introduction
-----------------------------------------
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset). 



Data
-----------------------------------------
The training data for this project are available here: 
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here: 
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

The data for this project come from this source: <http://groupware.les.inf.puc-rio.br/har>.

The data can also be downloaded using the following R scoprt:
```{r,cache=TRUE}
# essential library 
library(caret)
library(randomForest)
library(corrplot)

downloadFiles <- function(dataURL = "", destF = "t.csv")
{
    if(!file.exists(destF)){
        download.file(dataURL, destF, method="curl")
    }else{
        message("data already downloaded.")
    }
}
trainURL<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testURL <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
downloadFiles(trainURL, "pml-training.csv")
downloadFiles(testURL, "pml-testing.csv")
```
Load the data from csv files.
```{r,cache=TRUE}
train <- read.csv("./pml-training.csv")
test <- read.csv("./pml-testing.csv")
```
Check the structure of data and the number of each class in training set.
```{r,cache=TRUE}
dim(train)
table(train$classe)
```
Split training dataset into training and validation for evaluating model.
We did cross validation using createDataPartition().
This function guarantees diverse distribution in each column.
```{r,cache=TRUE}
set.seed(123456)
inTrain <- createDataPartition(train$classe, p = 3/4, list = FALSE)
trainingSet <- train[inTrain, ]
# create validation set for testing in sample error
validationSet <- train[-inTrain, ]
```


Fetures Slection for makeing model
-----------------------------------------
Check the near zero covariates(featrues).
```{r,cache=TRUE}
nzvMatrix <- nearZeroVar(trainingSet, saveMetrics = TRUE)
trainingSet_rmovedZero <- trainingSet[,!nzvMatrix$nzv]
```
Deal with missing value.
```{r,cache=TRUE}
# first option to handle missing value:
# remove columns, containing missing value over 50%, out of all data.
cntlength <- sapply(trainingSet_rmovedZero, function(x) {
    sum(!(is.na(x) | x == ""))
})
columnNA_frist <- names(cntlength[cntlength < 0.5 * length(trainingSet$classe)])

# second option to handle missing value:
# remove all columns, contating a missing value.
conditionColumnsNA <- apply(trainingSet_rmovedZero,2,function(x) table(is.na(x))[1]!=dim(trainingSet_rmovedZero)[1])   
columnNA_second <- names(trainingSet_rmovedZero)[conditionColumnsNA]
```

Discards unsueful covariates(feautres), beacuse these featrues are descriptive features.
So, we consider only numeric type of covariate from HAR sensor.
```{r,cache=TRUE}
descriptiveColumns <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", 
                 "cvtd_timestamp", "new_window", "num_window")
removeColumns <- c(descriptiveColumns, columnNA_second)
refinedTrainingSet <- trainingSet_rmovedZero[, !names(trainingSet_rmovedZero) %in% removeColumns]
```

Remove highly correlated covariates(features).
To comput correlation, make set without classes.
```{r,cache=TRUE}
corrM <- cor(subset(refinedTrainingSet, select=-c(classe)))
corrplot(corrM, method="circle",tl.cex=0.6)
```
Detect high correlation.
```{r,cache=TRUE}
highCorr <- findCorrelation(corrM, cutoff = .75)
```
To make concrete data set, combine two data, classe and data excluding high correlation of columns.
```{r,cache=TRUE}
removeHighCorrTrainSet <- cbind(classe=refinedTrainingSet$classe,refinedTrainingSet[,-highCorr])  
```


Making Model using Random Forest Algorithm
-----------------------------------------
```{r,cache=TRUE}
rfModel <- randomForest(classe ~ ., data = removeHighCorrTrainSet, importance = TRUE, ntrees = 10)
rfModel
plot(rfModel)
varImpPlot(rfModel,cex=.5)  
```


Testing Constructed Model
-----------------------------------------
We test model in term of in sample error and out of sample error.
```{r,cache=TRUE}
# training sample
ptraining <- predict(rfModel, removeHighCorrTrainSet)
print(confusionMatrix(ptraining, removeHighCorrTrainSet$classe))
# out of sample
pvalidation <- predict(rfModel, validationSet)
print(confusionMatrix(pvalidation, validationSet$classe))
```
The reason why out of sample error appared is probably noise called outlier or somehow it is beacuse our model is overffting.


Predicting test set, including no class data
-----------------------------------------
Test set prediction
```{r,cache=TRUE}
ptest <- predict(rfModel, test)
ptest
```

We then save the output to files based on instructions.
We can post it to the submission page.
```{r,cache=TRUE}
answers <- as.vector(ptest)
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}
pml_write_files(answers)
```

Conclusion
-----------------------------------------
In this project, we made predition model based on Human Actiity Recogniton (HAR) project. To build prediction model, we took several pre-processing steps that are zero variance features analysis, removing missing value and decriptive featrues, and deleting high correlation features. By doing that, we reduced computating time of bulding model and achieved high accruacy of prediction. Fortunately, RandomFrest-Model we made attained 100% accruacy in term of 20 samples in Test-set. However, 100% accruacy is not make sense. For precise evaluation, we need to get larger data set than now.



