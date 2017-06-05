# Practical Machine Learning: Course Project
Ming Wei Siw  

##Summary

This study is based on Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements by Ugulino et al.. Using the data acquired from their paper, this study attempts to predict the various body movements of the subjects based on accelerometers data from different parts of the body.

To cite the paper, 
"Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E)."

To classify correctly the actions of the various subjects, various prediction methods have been used, which are, random forest, boosting and model stacking. The most accurate method appears to be random forest, albeit requiring length computation time.

##Load Packages


```r
library(data.table)
library(caret)
library(randomForest)
library(parallel)
library(doParallel)
```

##Download and Load Data


```r
urltrain <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
#download training data
if(!file.exists("pml-training.csv")) 
  {download.file(urltrain, destfile = "pml-training.csv", method = "curl")}
#load data
data1 <- fread("pml-training.csv", header = T)
```

##Data Cleansing

Since the data also contains values not suitable for prediction, i.e., subjects names, they are removed from the training dataset. Also, there are variables in the data which contains too many empty observations to be useful for analysis, i.e., kurtosis, amplitude and skewness. To ease prediction, these are also removed from the dataset.


```r
#isolate variables of interest
data1.predictors <- data1[, -c(1:7)]
data1.predictors$classe <- as.factor(data1.predictors$classe)
#removing variables with NAs
na.sum.predictors <- apply(is.na(data1.predictors), 2, sum)
data1.predictors <- subset(data1.predictors, 
                              select = which(na.sum.predictors == 0))
#remove predictors with many empty entries
rem.kurt <- grepl("^kurtosis", names(data1.predictors))
data1.predictors <- data1.predictors[, !rem.kurt, with = F]
rem.skew <- grepl("^skewness", names(data1.predictors))
data1.predictors <- data1.predictors[, !rem.skew, with = F]
rem.max <- grepl("^max", names(data1.predictors))
data1.predictors <- data1.predictors[, !rem.max, with = F]
rem.min <- grepl("^min", names(data1.predictors))
data1.predictors <- data1.predictors[, !rem.min, with = F]
rem.amp <- grepl("^amplitude", names(data1.predictors))
data1.predictors <- data1.predictors[, !rem.amp, with = F]
```

##Split Data into Training, Test and Validation Set

The given dataset is split into three portions: training, testing and validation.


```r
set.seed(1007)
inBuild <- createDataPartition(data1.predictors$classe, p = 0.7, list = F)
validation <- data1.predictors[-inBuild, ]
build <- data1.predictors[inBuild, ]
inTrain <- createDataPartition(build$classe, p = 0.7, list = F)
training <- build[inTrain, ]
testing <- build[-inTrain, ]
```

##Configure Parallel Processing

To hasten calculations, parallel processing is used in this study.


```r
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
```

##Modelling

###Cross Validation

To enhance the prediction powers of these models, cross validation with 5 folds are repeated for 5 times.


```r
control <- trainControl(method = "repeatedcv", number = 5, repeats = 5,
                        allowParallel = T)
```

All the prediction methods used in this study are ensemble methods as they usually result in the highest accuracy while reducing both bias and variance.

###Random Forest

Random forest attemps to build many classification trees using subsets of both data and predictors. This form of resampling allows for low bias and variance, while preventing the problem of overfitting.


```r
m1 <- train(classe ~ ., method = "rf", data = training, trControl = control)
confusionMatrix(testing$classe, predict(m1, testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1168    2    1    0    0
##          B    5  788    4    0    0
##          C    0    4  712    2    0
##          D    0    0    2  669    4
##          E    0    0    2    3  752
## 
## Overall Statistics
##                                           
##                Accuracy : 0.993           
##                  95% CI : (0.9899, 0.9953)
##     No Information Rate : 0.2848          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9911          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9957   0.9924   0.9875   0.9926   0.9947
## Specificity            0.9990   0.9973   0.9982   0.9983   0.9985
## Pos Pred Value         0.9974   0.9887   0.9916   0.9911   0.9934
## Neg Pred Value         0.9983   0.9982   0.9974   0.9985   0.9988
## Prevalence             0.2848   0.1928   0.1751   0.1637   0.1836
## Detection Rate         0.2836   0.1914   0.1729   0.1625   0.1826
## Detection Prevalence   0.2844   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9974   0.9949   0.9929   0.9954   0.9966
```

###Boosting

Boosting combines weak predictors to form a strong predictor. In the process, weaker predictors are weighted and then combined to form the final predictor.


```r
m2 <- train(classe ~., method = "gbm", data = training, verbose = F,
            trControl = control)
confusionMatrix(testing$classe, predict(m2, testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1155   11    3    2    0
##          B   24  750   20    1    2
##          C    0   20  690    8    0
##          D    1    1   24  644    5
##          E    2   10    3   14  728
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9633          
##                  95% CI : (0.9571, 0.9689)
##     No Information Rate : 0.287           
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9536          
##  Mcnemar's Test P-Value : 0.0006431       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9772   0.9470   0.9324   0.9626   0.9905
## Specificity            0.9946   0.9859   0.9917   0.9910   0.9914
## Pos Pred Value         0.9863   0.9410   0.9610   0.9541   0.9617
## Neg Pred Value         0.9908   0.9874   0.9853   0.9927   0.9979
## Prevalence             0.2870   0.1923   0.1797   0.1625   0.1785
## Detection Rate         0.2805   0.1821   0.1676   0.1564   0.1768
## Detection Prevalence   0.2844   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9859   0.9664   0.9621   0.9768   0.9910
```

###Bagging

Bagging, or Bootstrap Aggregating, creates bootstrap samples from the training set and fits a model using each resample. By averaging the fits from each samples, bagging also reduces bias and variance while avoiding overfitting.


```r
m3 <- train(classe ~., method = "treebag", data = training, 
            trControl = control)
confusionMatrix(testing$classe, predict(m3, testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1162    5    2    2    0
##          B    7  779   11    0    0
##          C    0   10  706    2    0
##          D    2    0    7  663    3
##          E    3    1    3    4  746
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9849          
##                  95% CI : (0.9807, 0.9884)
##     No Information Rate : 0.2851          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.981           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9898   0.9799   0.9684   0.9881   0.9960
## Specificity            0.9969   0.9946   0.9965   0.9965   0.9967
## Pos Pred Value         0.9923   0.9774   0.9833   0.9822   0.9855
## Neg Pred Value         0.9959   0.9952   0.9932   0.9977   0.9991
## Prevalence             0.2851   0.1931   0.1770   0.1629   0.1819
## Detection Rate         0.2822   0.1892   0.1714   0.1610   0.1812
## Detection Prevalence   0.2844   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9934   0.9872   0.9825   0.9923   0.9964
```

###Model Stacking With Validation Data

The three methods above are used together to reduce bias and variance. Moreover, model stacking also reduces the chance of overfitting. The validation dataset are used here to check for the accuracy of the prediction algorithm.


```r
stack.df <- data.frame(rf = predict(m1, testing), gbm = predict(m2, testing), 
                       bag = predict(m3, testing), classe = testing$classe)
comb.fit1 <- train(classe ~., data = stack.df, method = "rf", 
                   trControl = control)
pred.va.rf <- predict(m1, validation)
pred.va.boost <- predict(m2, validation)
pred.va.bag <- predict(m3, validation)
stack.df.va <- data.frame(rf = pred.va.rf, gbm = pred.va.boost,
                          bag = pred.va.bag, classe = validation$classe)
confusionMatrix(stack.df.va$classe, predict(comb.fit1, stack.df.va))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1669    4    1    0    0
##          B   13 1123    3    0    0
##          C    0    8 1015    3    0
##          D    0    0    9  953    2
##          E    2    0    2    8 1070
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9907         
##                  95% CI : (0.9879, 0.993)
##     No Information Rate : 0.2862         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9882         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9911   0.9894   0.9854   0.9886   0.9981
## Specificity            0.9988   0.9966   0.9977   0.9978   0.9975
## Pos Pred Value         0.9970   0.9860   0.9893   0.9886   0.9889
## Neg Pred Value         0.9964   0.9975   0.9969   0.9978   0.9996
## Prevalence             0.2862   0.1929   0.1750   0.1638   0.1822
## Detection Rate         0.2836   0.1908   0.1725   0.1619   0.1818
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9950   0.9930   0.9916   0.9932   0.9978
```

##Prediction With Testing Dataset

###Download and Load Data


```r
urltest <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
#download testing data
if(!file.exists("pml-testing.csv")) 
  {download.file(urltest, destfile = "pml-testing.csv", method = "curl")}
#load data
data2 <- fread("pml-testing.csv", header = T)
```

###Prediction with Stacked Model


```r
stack.pred <- data.frame(rf = predict(m1, data2), gbm = predict(m2, data2),
                         bag = predict(m3, data2))
pred <- predict(comb.fit1, stack.pred)
```

The predicted classes of activities are B, A, B, A, A, E, D, B, A, A, B, C, B, A, E, E, A, B, B, B.

##De-Register Parallel Processing

Deregistering parallel processing will end the use of multithreading for R.


```r
stopCluster(cluster)
registerDoSEQ()
```

##Conclusion

In short, all the models works very well in predicting the different types of movements made by the subjects, each with an accuracy greater than 95%. If the sample where these algorithms is applied is somewhat similar to this dataset, i.e., no real outliers or no false entry, the out-of-sample error should be rather low, perhaps at around 5% or less. 
