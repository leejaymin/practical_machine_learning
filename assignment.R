
# essential library 
library(caret)
library(randomForest)
library(corrplot)

#Load the data from csv files
train <- read.csv("./pml-training.csv")
test <- read.csv("./pml-testing.csv")
# check the structure of data and the number of each class in training set
dim(train)
table(train$classe)

set.seed(123456)
inTrain <- createDataPartition(train$classe, p = 3/4, list = FALSE)
trainingSet <- train[inTrain, ]
# create validation set for testing in sample error
validationSet <- train[-inTrain, ]


#-- Feature Selection --
# check the near zero covariates(featrues)
nzvMatrix <- nearZeroVar(trainingSet, saveMetrics = TRUE)
trainingSet_rmovedZero <- trainingSet[,!nzvMatrix$nzv]

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


# discards unsueful covariates(feautres)
# beacuse these featrues are descriptive features 
# we consider only numeric type of covariate from HAR sensor
descriptiveColumns <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", 
                 "cvtd_timestamp", "new_window", "num_window")
removeColumns <- c(descriptiveColumns, columnNA_second)
refinedTrainingSet <- trainingSet_rmovedZero[, !names(trainingSet_rmovedZero) %in% removeColumns]


# remove highly correlated covariates(features)
# to comput correlation, make set without classes
corrM <- cor(subset(refinedTrainingSet, select=-c(classe)))
corrplot(corrM, method="circle",tl.cex=0.6)
# detect high correlation
highCorr <- findCorrelation(corrM, cutoff = .75)
# to make concrete data set, combine two data, classe and data excluding high correlation of columns
removeHighCorrTrainSet <- cbind(classe=refinedTrainingSet$classe,refinedTrainingSet[,-highCorr])  


#--- model train
rfModel <- randomForest(classe ~ ., data = removeHighCorrTrainSet, importance = TRUE, ntrees = 10)
rfModel

#--- model validation
# training sample
ptraining <- predict(rfModel, removeHighCorrTrainSet)
print(confusionMatrix(ptraining, removeHighCorrTrainSet$classe))
# out of sample
pvalidation <- predict(rfModel, validationSet)
print(confusionMatrix(pvalidation, validationSet$classe))


# test set prediction
ptest <- predict(rfModel, test)
ptest

# prediction assignment submission: instructions
#answers = rep("A", 20)
answers <- as.vector(ptest)
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}
pml_write_files(answers)
