setwd("C:/Users/preek/Desktop/SDSU MIS/MIS 620")
h1B <- read.csv("MasterH1BDataset.csv")
h1B <- na.omit(h1B)
str(h1B)
library(caret)
#h1b <- subset(h1B, select = -c(EMPLOYER_NAME))
summaryh1b
##adapted from https://topepo.github.io/caret 
##Data Splitting
#library(caret)
library("randomForest")
library("party")
library("dplyr")
library("tidyr")
library("sqldf")
library("caret")
library("glmnet")
library("car")
library("ROCR")
##library("gbm")
#always set seed to make code reproducable
#whenever anything has randomisation invloved identical not close
set.seed(3456)
#Perserve class distribution and split 80% and 20% subsets
#want vector
#h1B$CASE_STATUS <- factor(c( "CERTIFIED","WITHDRAWN"))
#levels(CASE_STATUS) <- c("c", "a", "b")
#X = c(1,2,3,4)
#levels(h1B$CASE_STATUS) <- list("CERTIFIED" = c("CERTIFIED", "CERTIFIEDWITHDRAWN"), "WITHDRAWN" = c("WITHDRAWN", "DENIED"))
str(h1B$CASE_STATUS)
#h1B$PW_SOURCE_OTHER <- dummyVars( ~., data = h1B)
#h1bclean <- na.omit(h1b)
#h1bclean$EMPLOYER_NAME <- as.numeric(h1bclean$EMPLOYER_NAME)
h1B$EMPLOYER_NAME <- NULL
h1B <- subset(h1B, select = -c(WORKSITE_POSTAL_CODE))
#trainIndex <- createDataPartition(h1B$CASE_STATUS, p = .8, 
                                  #list = FALSE, 
h1bnull <- subset(h1B, h1B$CASE_STATUS != "WITHDRAWN")
h1bnull$CASE_STATUS[h1bnull$CASE_STATUS == "CERTIFIEDWITHDRAWN"] <- "CERTIFIED"
h1bdrop <- h1bnull
h1bdrop$CASE_STATUS <- droplevels(h1bnull$CASE_STATUS)    
str(h1bdrop)

#h1bwow <- subset(h1bclean, h1bclean$CASE_STATUS != "WITHDRAWN")
#h1bwow$CASE_STATUS[h1bwow$CASE_STATUS == "CERTIFIEDWITHDRAWN"] <- "CERTIFIED"
#h1bdrop <- h1bwow
#h1bdrop$CASE_STATUS <- droplevels(h1bwow$CASE_STATUS)
#times = 1)
#head(trainIndex)
#create datapartiton wats label iris..11
#cor.test(h1B$PW_SOURCE_OTHER, h1B$SOC_NAME)
#create training set subset (80%)
#h1BTrain <- h1B[ trainIndex,]

#create training set subset (20%)
#h1BTest  <- h1B[-trainIndex,]

#table(h1BTrain$CASE_STATUS)
#table(h1BTest$CASE_STATUS)

#str(etitanic) #notice factors
head(h1bdrop)

#model them as dummy variables
#head(model.matrix(CASE_STATUS ~ . - EMPLOYER_NAME - WORKSITE_POSTAL_CODE, data = h1B))
head(model.matrix(CASE_STATUS ~ . , data = h1bdrop))
#to convert to dummy variables use dummyVars to create a model
#dummies <- dummyVars(survived ~ ., data = etitanic, fullRank =TRUE)
#dummies <- dummyVars( ~ . - EMPLOYER_NAME - WORKSITE_POSTAL_CODE, data = h1B)
dummies <- dummyVars( ~ ., data = h1bdrop)


#dont use full rank
head(h1bdrop)
#created pclass 1,2,3rd..

#h1B[h1B$CASE_STATUS == C(CERTIFIED)]
#use dummy vars to survived,dummy is model..use predict toapply to new data

#apply that model with the predict function make sure and save to new data.frame
dummies.h1bdrop <- head(predict(dummies, newdata = h1bdrop))
class(dummies)

#using the AppliedPredictiveModeling package for the book
library(AppliedPredictiveModeling)
#CASE_STATUS <- factor(c("CERTIFIED","WITHDRAWN"))
#levels(CASE_STATUS) <- c("c", "a", "b")
#X = c(1,2,3,4)
#levels(CASE_STATUS) <- list("CERTIFIED" = c("CERTIFIED", "CERTIFIEDWITHDRAWN"), "WITHDRAWN" = c("WITHDRAWN", "DENIED"))
#CASE_STATUS
#we can do all preprocessing in one step!
#trying to predict 8 th class omit last 8 th column
#leaving out class variable
#pp_hpc <- preProcess(h1B[, -8,-26], 
                     #method = c("center", "scale", "BoxCox"))
#pp_hpc #can see results of processing

pp_hpc1 <- preProcess(h1bclean[, ], 
                     method = c("center", "scale", "BoxCox","nzv"))
pp_hpc1 #can see results of processing

#now we juse need to apply the processing model to the data
#transformed <- predict(pp_hpc, newdata = h1B[, -8,-26])
#head(transformed)

transformed1 <- predict(pp_hpc1, newdata = h1bclean[, ])
head(transformed1)


#numpending very low variance
mean(h1bdrop$NumPending == 0) #76%=0

#feature reduction using PCA

#pp_no_pca <- preProcess(h1B[, -8, -26], method = c("pca"))
#pp_no_pca

pp_no_pca1 <- preProcess(h1bclean[, ], method = c("pca"))
pp_no_pca1
#varImp(pp_no_pca1)
#apply model to data and see new subset and processed variables
#head(predict(pp_no_pca, newdata = h1B[ -8, -26]))
#NOT MUCH IMPUTATION...DATA SET NOT MUCH IMPUTATION  IN THIS COURSE
#interpret--1pc contains most inof,each subset less n less..find least amt explains most amt
#pp_no_nzv <- preProcess() add ("pca", "knnImpute")

head(predict(pp_no_pca1, newdata = h1bclean[ , ]))
finaldata <- (predict(pp_no_pca1, newdata = h1bclean[ , ]))
#lets fit PCA making sure to scale
h1b.pca <- prcomp(dummies.h1b, scale=TRUE) 

#view scaling centers and SD
hitters.pca$center
hitters.pca$scale

#view the pca loadings one PC for each 19 variables
finaldata$rotation

#view biplot of first two components
biplot(finaldata, scale=0)

#variance explained by each component, squaring standard deviation 
pca.var<- finaldata$sdev^2

#proportion of variance explained
pve<- pca.var/ sum(pca.var)

#scree plot, variance explained by component
plot(pve, xlab="PCA", ylab="Prop of Variance Explained", ylim=c(0,1), type='b')

#cumulative variance explained
#scree plot, variance explained by component
plot(cumsum(pve), xlab="PCA", ylab="Cumulative Prop of Variance Explained", ylim=c(0,1), type='b')

#grabe the first five PCs
hitters.pca$x[,1:5]

#shorter version of summarizing variance
summary(hitters.pca)
plot(finaldata)
#ome parameters to control the sampling during parameter tuning and testing
#5 fold crossvalidation, using 5-folds instead of 10 to reduce computation time in class demo, use 10 and with more computation to spare use
#repeated cv
ctrl <- trainControl(method="cv", number=2,
                     classProbs=TRUE,
                     #function used to measure performance
                     summaryFunction = twoClassSummary, #multiClassSummary for non binary
                     allowParallel =  TRUE) #am disabling allowParallel because of bug in caret 6.0-77
#example models are small enough that parallel not important but set to 
#filter(h1B, CASE_STATUS==CERTIFIED)
#h1b <-filter(h1B$CASE_STATUS %in% c('CERTIFIED','DENIED'))
#only females
#select(h1B, CASE_STATUS ==CERTIFIED)
#filter(hsb2, female==1 & write >50) #only females
#CASE_STATUS<- factor(CASE_STATUS)
#CASE_STATUS <- (CERTIFIED=c("CERTIFIED", "CERTIFIEDWITHDRAWN"), WITHDRAWN=c("DENIED", "WITHDRAWN"))
#CASE_STATUS 

trainIndex <- createDataPartition(finaldata$CASE_STATUS, p = .8, 
                                  list = FALSE, 
                                  times = 1)
head(trainIndex)
#create datapartiton wats label iris..11
#cor.test(h1B$PW_SOURCE_OTHER, h1B$SOC_NAME)
h1BTrain <- h1bdrop[ trainIndex,]

#create training set subset (20%)
h1BTest  <- h1bdrop[-trainIndex,]
table(h1BTrain$CASE_STATUS)
table(h1BTest$CASE_STATUS)


library(DMwR)
#hybrid both up and down
set.seed(9560)
#smote_train <- SMOTE(CASE_STATUS ~ .- EMPLOYER_NAME - WORKSITE_POSTAL_CODE, data  = h1BTrain)  
smote_train <- SMOTE(CASE_STATUS ~ ., data  = h1BTrain)   
table(smote_train$CASE_STATUS) 

library(ROSE)
set.seed(9560)
#rose_train <- ROSE(CASE_STATUS ~.- EMPLOYER_NAME - WORKSITE_POSTAL_CODE, data = h1BTrain)$data 
rose_train <- ROSE(CASE_STATUS ~ ., data = h1BTrain)$data
#rose_train <- ROSE(CASE_STATUS ~ ., data = dummies.h1B)$data
table(rose_train$CASE_STATUS)


#Logistic regression:
##logistic regression
set.seed(199)#ALWAYS USE same SEED ACROSS trains to ensure identical cv folds
#h1B.log <-  train(CASE_STATUS ~ .- EMPLOYER_NAME - WORKSITE_POSTAL_CODE - PW_SOURCE_OTHER, data=rose_train, method="glm", family="binomial", metric="ROC", trControl=ctrl)
#h1B.log <-  train(CASE_STATUS ~ .- EMPLOYER_NAME - WORKSITE_POSTAL_CODE - SOC_NAME - PW_SOURCE_OTHER, data=rose_train, method="glm", family="binomial", metric="ROC", trControl=ctrl)
h1B.log <-  train(CASE_STATUS ~ ., data=rose_train, method="glm", family="binomial", metric="ROC", trControl=ctrl)

summary(h1B.log)
varImp(h1B.log)
getTrainPerf(h1B.log)
h1B.log 
#calculate resampled accuracy/confusion matrix using extracted predictions from resampling
confusionMatrix(h1B.log$pred$pred, h1B.log$pred$obs) #take averages
p.log<- predict(h1B.log,h1BTest)
confusionMatrix(h1B.log,h1BTest)

#NAIVE BAYES:
##Naive Bayes
modelLookup("nb") #we have some paramters to tune such as laplace correction
set.seed(192)
library(MLmetrics)
#h1B.nb <- train(CASE_STATUS ~ .-EMPLOYER_NAME - WORKSITE_POSTAL_CODE, data = h1BTrain,
                #trControl = ctrl,
                #metric = "ROC", #using AUC to find best performing parameters
                #method = "nb")
h1B.nb <- train(CASE_STATUS ~ ., data = rose_train,
                trControl = ctrl,
                metric = "ROC", #using AUC to find best performing parameters
                method = "nb")
h1B.nb
getTrainPerf(h1B.nb)
varImp(h1B.nb)
plot(h1B.nb)
h1B.nb<- predict(h1B.nb,h1BTest)
#confusionMatrix(p.rpart,y.test) #calc accuracies with confuction matrix on test set


#random forest approach to many classification models created and voted on
#less prone to ovrefitting and used on large datasets
library(randomForest)
set.seed(192)
modelLookup("rf")
#h1B.rf <- train(CASE_STATUS ~ .-EMPLOYER_NAME - WORKSITE_POSTAL_CODE, data = h1BTrain,
              #trControl = ctrl,
              #metric = "ROC", #using AUC to find best performing parameters
              #tuneLength=9,
              #method = c("rf") )
h1B.rf <- train(CASE_STATUS ~ ., data = rose_train,
                trControl = ctrl,
                metric = "ROC", #using AUC to find best performing parameters
                tuneLength=9,
                method = c("rf") )
h1B.rf

p.rf<- predict(h1B.rf,h1BTest)
confusionMatrix(h1B.rf,h1BTest)

##linear discriminant analysis
set.seed(199)
h1B.lda <-  train(CASE_STATUS ~ .,data = rose_train, method="lda", metric="ROC", trControl=ctrl)
#h1B.lda <- train(CASE_STATUS ~ . - EMPLOYER_NAME - WORKSITE_POSTAL_CODE, data = smote_train, method="lda", metric="ROC", trControl=ctrl)
h1B.lda
varImp(h1B.lda)
confusionMatrix(h1B.lda$pred$pred, h1B.lda$pred$obs) #take averages
p.rf<- predict(h1B.lda,h1BTest)
confusionMatrix(h1B.lda,h1BTest)

##quadratic distriminant analysis
set.seed(199)
h1B.qda <-  train(CASE_STATUS ~ ., data = rose_train, method="qda", metric="ROC", trControl=ctrl)
#h1B.qda <-  train(CASE_STATUS ~ h1b.emplname + h1b.Pos + h1b.PW + CASE_SUBMITTED_MONTH + CASE_SUBMITTED_DAY + CASE_SUBMITTED_YEAR + DECISION_DAY + DECISION_MONTH + DECISION_YEAR + VISA_CLASS +  EMPLOYER_STATE + EMPLOYER_COUNTRY + SOC_NAME + NAICS_CODE + TOTAL_WORKERS + FULL_TIME_POSITION + PREVAILING_WAGE + PW_UNIT_OF_PAY + PW_SOURCE + PW_SOURCE_YEAR + WAGE_RATE_OF_PAY_FROM + WAGE_RATE_OF_PAY_TO + WAGE_UNIT_OF_PAY + H.1B_DEPENDENT + WILLFUL_VIOLATOR + WORKSITE_STATE,data = h1B.train, method="qda", metric="ROC", trControl=ctrl)
h1B.qda
p.rf<- predict(h1B.qda,h1BTest)
confusionMatrix(h1B.qda,h1BTest)
getTrainPerf(h1B.qda)
varImp(h1B.qda)

#k nearest neighbors classification
set.seed(199) 
kvalues <- expand.grid(k=1:20)

h1B.knn <-  train(CASE_STATUS ~ ., data = rose_train, method="knn", metric="ROC", trControl=ctrl, tuneLength=10) #let caret decide 10 best parameters to search
h1B.knn
plot(h1B.knn)
getTrainPerf(h1B.knn)
p.rf<- predict(h1B.knn,h1BTest)
confusionMatrix(h1B.knn,h1BTest)
confusionMatrix(h1B.knn$pred$pred, h1B.knn$pred$obs) #make sure to select resamples only for optimal parameter of K

#really need test set to get more accurate idea of accuracy when their is a rare class
#can either use model on cross validation of complete training data or hold out test set

#lets compare all resampling approaches
h1B.models <- list("logit"=h1B.log, "lda"=h1B.lda, "qda"=h1B.qda,
                   "knn"=h1B.knn)
d.resamples = resamples(h1B.models)


#plot performance comparisons
bwplot(h1B.resamples, metric="ROC") 
bwplot(h1B.resamples, metric="Sens") #predicting default dependant on threshold
bwplot(h1B.resamples, metric="Spec") 

#calculate ROC curves on resampled data

h1B.log.roc<- roc(response= h1B.log$pred$obs, predictor=h1B.log$pred$Yes)
h1B.lda.roc<- roc(response= h1B.lda$pred$obs, predictor=h1B.lda$pred$Yes)
h1B.qda.roc<- roc(response= h1B.qda$pred$obs, predictor=h1B.qda$pred$Yes)
#when model has parameters make sure to select final parameter value
h1B.knn.roc<- roc(response= h1B.knn$pred[h1B.knn$pred$k==23,]$obs, predictor=h1B.knn$pred[h1B.knn$pred$k==23,]$Yes) 

#build to combined ROC plot with resampled ROC curves
plot(h1B.log.roc, legacy.axes=T)
plot(h1B.lda.roc, add=T, col="Blue")
plot(h1B.qda.roc, add=T, col="Green")
plot(h1B.knn.roc, add=T, col="Red")
legend(x=.2, y=.7, legend=c("Logit", "LDA", "QDA", "KNN"), col=c("black","blue","green","red"),lty=1)

#logit looks like the best choice its most parsimonious and equal ROC to LDA, QDA and KNN

#lets identify a more optimal cut-off (current resampled confusion matrix), low sensitivity
confusionMatrix(h1B.log$pred$pred, h1B.log$pred$obs)

#extract threshold from roc curve  get threshold at coordinates top left most corner
h1B.log.Thresh<- coords(h1B.log.roc, x="best", best.method="closest.topleft")
h1B.log.Thresh #sensitivity increases to 88% by reducing threshold to .0396 from .5

#lets make new predictions with this cut-off and recalculate confusion matrix
h1B.log.newpreds <- factor(ifelse(h1B.log$pred$Yes > h1B.log.Thresh[1], "Yes", "No"))

#recalculate confusion matrix with new cut off predictions
confusionMatrix(h1B.log.newpreds, h1B.log$pred$obs)

### TEST DATA PERFORMANCE
#lets see how this cut off works on the test data
#predict probabilities on test set with log trained model
test.pred.prob <- predict(h1B.log, h1BTest, type="prob")

test.pred.class <- predict(h1B.log, h1BTest) #predict classes with default .5 cutoff

#calculate performance with confusion matrix
confusionMatrix(test.pred.class, h1BTest$CASE_STATUS)

#let draw ROC curve of training and test performance of logit model
test.log.roc<- roc(response= h1BTest$CASE_STATUS, predictor=test.pred.prob[[1]]) #assumes postive class Yes is reference level
plot(test.log.roc, legacy.axes=T)
plot(h1B.log.roc, add=T, col="blue")
legend(x=.2, y=.7, legend=c("Test Logit", "Train Logit"), col=c("black", "blue"),lty=1)

#test performance slightly lower than resample
auc(test.log.roc)
auc(h1B.log.roc)

#calculate test confusion matrix using thresholds from resampled data
test.pred.class.newthresh <- factor(ifelse(test.pred.prob[[1]] > h1B.log.Thresh[1], "Yes", "No"))

#recalculate confusion matrix with new cut off predictions
confusionMatrix(test.pred.class.newthresh, h1BTest$CASE_STATUS)


#you have to adjust thresholds when dealing with unbalanced data
#don't ignore cost of FPR or FNR, falsely accusing may be expensive



###BONUS PLOTS to calibrate threshold LIFT AND CALIBRATION
#create a lift chart of logit test probabilities against
test.lift <- lift(h1BTest$CASE_STATUS ~ test.pred.prob[[1]]) #Lift
plot(test.lift)

test.cal <- calibration(h1BTest$CASE_STATUS ~ test.pred.prob[[1]]) #Calibration 
plot(test.cal)


pp.thresh <- glm(h1BTest$CASE_STATUS ~ test.pred.prob[[1]], family="binomial")


predict(pp.thresh, as.data.frame(test.pred.prob[[1]]))

library(ada)
set.seed(192)
#boosted decision trees
#using dummy codeds because this function internally does it and its better to handle it yourself (i.e., less error prone)
modelLookup("ada")
h1B.ada <- train(CASE_STATUS ~ ., data = rose_train,
               trControl = ctrl,
               metric = "ROC", #using AUC to find best performing parameters
               method = "ada")


h1B.ada
plot(h1B.ada)
p.ada<- predict(h1B.ada,h1BTest)
confusionMatrix(h1B.ada,h1BTest)



#compare the performance of all models trained today
#rValues <- resamples(list(rpart=m.rpart, nb=h1B.nb, log=h1B.glm, bag=m.bag, boost=h1B.ada, rf=h1B.rf))

bwplot(rValues, metric="ROC")
bwplot(rValues, metric="Sens") #Sensitvity
bwplot(rValues, metric="Spec")


set.seed(5627) #set.seed not completely sufficient when multicore
h1B.gam <- train(CASE_STATUS ~ ., data = rose_train, 
                  method = "gamSpline",tuneLength=2,
                  metric = "ROC",
                  trControl = ctrl)
confusionMatrix(predict(h1B.gam,h1BTest), h1BTest$CASE_STATUS)
p.rf<- predict(h1B.gam,h1BTest)
confusionMatrix(h1B.gam,h1BTest)
#pre processing time. Typically you will need to first convert to dummy codes to complete this step
#cant calculate correlation between variables for example otherwise


library(mice) #imputation package will discuss later
library(VIM)

#check pattern of missing data
md.pattern(h1bdrop)

#we have none in this sample set, but this will never be the case in real life
#lets randomly remove some to see how to interpret missing data plots/patterns

#set.seed(192)
#r.col <- sample(1:16, size=100, replace=T)
#r.row <- sample(1:1000, size=100, replace =T)

h1bdrop.missing <- h1B

#for simplicity we will loop thoguh data frame removing columns and rows (ideally we vectorize)
#for (n in 1:100)
#{
  
  #h1B.missing[r.row[n], r.col[n]] <- NA
  
  
#}

#now lets look at missing data pattern
md.pattern(h1B.missing)
aggr_plot <- aggr(h1B.missing, col=c('navyblue','red'), 
                  numbers=TRUE, sortVars=TRUE, labels=names(h1B.missing))
#clearly missing at random

#lets detach mice and VIM, caret can be sensitive to conflicts accross loaded packages
#try to keep libraries load to a minimum when modeling
unloadNamespace("mice")
unloadNamespace("VIM")


#will walk through basic imputing
set.seed(192)
#caret has preprocess function - we are imputing missing data with bag (ensemble decision trees), scaling and centering, and filtering out
#highly correlated predictors
#x.prepmodel <- preProcess(h1B.dummy, method=c("bagImpute", "scale", "center", "zv", "corr"))

#x.prepmodel$
  
  
  
  #apply pre processing model to training/test data
  x.prep <- predict(x.prepmodel, x.dummy)

str(x.prep)

