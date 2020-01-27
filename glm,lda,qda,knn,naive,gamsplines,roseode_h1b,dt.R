setwd("C:/Users/preek/Desktop/SDSU MIS/MIS 620") 
library(caret)
h1b <- read.csv("MasterH1BDataset.csv")
#worksite postal code removed, employer name converted to numeric
h1bclean <- na.omit(h1b)
#h1bclean$EMPLOYER_NAME <- as.numeric(h1bclean$EMPLOYER_NAME)
h1bclean$EMPLOYER_NAME <- NULL
h1bclean <- subset(h1bclean, select = -c(WORKSITE_POSTAL_CODE))
trainIndex <- createDataPartition(h1bclean$CASE_STATUS, p=.7, list=F)   
h1b.train <- h1bclean[trainIndex,]
h1b.test <-  h1bclean[-trainIndex,]
str(h1bclean$CASE_SUBMITTED_YEAR)
summary(h1bclean$CASE_SUBMITTED_YEAR)
ctrl <- trainControl(method="cv", number=3, classProbs=TRUE)
#h1b.rpart <- train(CASE_STATUS ~ ., data=h1b.train, trControl = ctrl, metric="ROC", method="rpart")
#rpart.plot(h1b.rpart$finalModel)

h1bwow <- subset(h1bclean, h1bclean$CASE_STATUS != "WITHDRAWN")
h1bwow$CASE_STATUS[h1bwow$CASE_STATUS == "CERTIFIEDWITHDRAWN"] <- "CERTIFIED"
h1bdrop <- h1bwow
h1bdrop$CASE_STATUS <- droplevels(h1bwow$CASE_STATUS)
library(DMwR)
smote.train <- SMOTE(CASE_STATUS ~ ., data = h1bdrop)
table(smote.train$CASE_STATUS)
library(ROSE)
rose.train <- ROSE(CASE_STATUS ~ ., data = h1bdrop)$data
table(rose.train$CASE_STATUS)

set.seed(199)#ALWAYS USE same SEED ACROSS trains to ensure identical cv folds
#h1B.log <-  train(CASE_STATUS ~ .- EMPLOYER_NAME - WORKSITE_POSTAL_CODE - PW_SOURCE_OTHER, data=rose_train, method="glm", family="binomial", metric="ROC", trControl=ctrl)
#h1B.log <-  train(CASE_STATUS ~ .- EMPLOYER_NAME - WORKSITE_POSTAL_CODE - SOC_NAME - PW_SOURCE_OTHER, data=rose_train, method="glm", family="binomial", metric="ROC", trControl=ctrl)
#h1B.log <-  train(CASE_STATUS ~ . - PW_SOURCE_OTHER, data=rose.train, method="glm", family="binomial", metric="ROC", trControl=ctrl)
h1B.log <-  train(CASE_STATUS ~ ., data=rose.train, method="glm", family="binomial", metric="ROC",preProcess = c("scale", "center"), trControl=ctrl)

summary(h1B.log)
varImp(h1B.log)
getTrainPerf(h1B.log)
h1B.log 
h1B.nb<- predict(h1B.log,h1b.test)
confusionMatrix(h1B.log,h1b.test)
#calculate resampled accuracy/confusion matrix using extracted predictions from resampling
confusionMatrix(h1B.log$pred$pred, h1B.log$pred$obs) #take averages

#NAIVE BAYES:
##Naive Bayes
modelLookup("nb") #we have some paramters to tune such as laplace correction
set.seed(192)
library(MLmetrics)
#h1B.nb <- train(CASE_STATUS ~ .-EMPLOYER_NAME - WORKSITE_POSTAL_CODE, data = h1BTrain,
                #trControl = ctrl,
                #metric = "ROC", #using AUC to find best performing parameters
                #method = "nb")
h1B.nb <- train(CASE_STATUS ~ ., data = rose.train,
                trControl = ctrl,
                metric = "ROC", #using AUC to find best performing parameters
                method = "nb")
h1B.nb
getTrainPerf(h1B.nb)
varImp(h1B.nb)
plot(h1B.nb)
h1B.nb<- predict(h1B.nb,h1b.test)
confusionMatrix(h1B.nb,h1b.test) #calc accuracies with confuction matrix on test set


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
h1B.rf <- train(CASE_STATUS ~ ., data = rose.train,
                trControl = ctrl,
                metric = "ROC", #using AUC to find best performing parameters
                tuneLength=9,
                method = c("rf") )
h1B.rf

p.rf<- predict(h1B.rf,h1BTest)
confusionMatrix(h1B.rf,h1BTest)

##linear discriminant analysis
set.seed(199)
#h1B.lda <-  train(CASE_STATUS ~ .,data = h1B.train, method="lda", metric="ROC", trControl=ctrl)
#h1B.lda <- train(CASE_STATUS ~ . - EMPLOYER_NAME - WORKSITE_POSTAL_CODE, data = smote_train, method="lda", metric="ROC", trControl=ctrl)
h1B.lda <- train(CASE_STATUS ~ ., data = rose.train, method="lda", metric="ROC", trControl=ctrl)
h1B.lda
varImp(h1B.lda)
confusionMatrix(h1B.lda$pred$pred, h1B.lda$pred$obs) #take averages
h1B.nb<- predict(h1B.lda,h1b.test)
confusionMatrix(h1B.lda,h1b.test)

library(ada)
set.seed(192)
#boosted decision trees
#using dummy codeds because this function internally does it and its better to handle it yourself (i.e., less error prone)
modelLookup("ada")
#h1B.ada <- train(CASE_STATUS ~ . - EMPLOYER_NAME - WORKSITE_POSTAL_CODE, data = smote_train,
                 #trControl = ctrl,
                 #metric = "ROC", #using AUC to find best performing parameters
                 #method = "ada")
h1B.ada <- train(CASE_STATUS ~ ., data = rose.train,
                 trControl = ctrl,
                 metric = "ROC", #using AUC to find best performing parameters
                 method = "ada")

h1B.ada
plot(h1B.ada)
p.ada<- predict(h1B.ada,h1b.test)
confusionMatrix(h1B.ada,h1b.test)



#compare the performance of all models trained today
rValues <- resamples(list(rpart=m.rpart, nb=h1B.nb, log=h1B.glm, rf=h1B.rf)) #bag=m.bag, boost=h1B.ada,))

bwplot(rValues, metric="ROC")
bwplot(rValues, metric="Sens") #Sensitvity
bwplot(rValues, metric="Spec")


set.seed(5627) #set.seed not completely sufficient when multicore
#h1B.gam <- train(CASE_STATUS ~ . - EMPLOYER_NAME - WORKSITE_POSTAL_CODE, data = rose_train, 
                 #method = "gamSpline",tuneLength=2,
                 #metric = "ROC",
                 #trControl = ctrl)
h1B.gam <- train(CASE_STATUS ~ ., data = rose_train, 
                 method = "gamSpline",tuneLength=2,
                 metric = "ROC",
                 trControl = ctrl)
p.gam <- predict(h1B.gam,h1b.test)
confusionMatrix(p.gam, h1b.test)
confusionMatrix(p.gam, h1b.test$CASE_STATUS)

##quadratic distriminant analysis
set.seed(199)
h1B.qda <-  train(CASE_STATUS ~ ., data = rose_train, method="qda", metric="ROC", trControl=ctrl)
#h1B.qda <-  train(CASE_STATUS ~ h1b.emplname + h1b.Pos + h1b.PW + CASE_SUBMITTED_MONTH + CASE_SUBMITTED_DAY + CASE_SUBMITTED_YEAR + DECISION_DAY + DECISION_MONTH + DECISION_YEAR + VISA_CLASS +  EMPLOYER_STATE + EMPLOYER_COUNTRY + SOC_NAME + NAICS_CODE + TOTAL_WORKERS + FULL_TIME_POSITION + PREVAILING_WAGE + PW_UNIT_OF_PAY + PW_SOURCE + PW_SOURCE_YEAR + WAGE_RATE_OF_PAY_FROM + WAGE_RATE_OF_PAY_TO + WAGE_UNIT_OF_PAY + H.1B_DEPENDENT + WILLFUL_VIOLATOR + WORKSITE_STATE,data = h1B.train, method="qda", metric="ROC", trControl=ctrl)
h1B.qda
getTrainPerf(h1B.qda)
varImp(h1B.qda)
p.qda <- predict(h1B.qda,h1b.test)
confusionMatrix(p.qda, h1b.test)

#k nearest neighbors classification
set.seed(199) 
kvalues <- expand.grid(k=1:20)

#h1B.knn <-  train(CASE_STATUS ~ . - EMPLOYER_NAME - WORKSITE_POSTAL_CODE, data = h1BTrain, method="knn", metric="ROC", trControl=ctrl, tuneLength=10) #let caret decide 10 best parameters to search
h1B.knn <-  train(CASE_STATUS ~ ., data = rose_train, method="knn", metric="ROC", trControl=ctrl, tuneLength=10) #let caret decide 10 best parameters to search
h1B.knn
plot(h1B.knn)
getTrainPerf(h1B.knn)
p.knn <- predict(p.knn,h1b.test)
confusionMatrix(p.knn, h1b.test)
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
test.pred.prob <- predict(h1B.log, h1b.test, type="prob")

test.pred.class <- predict(h1B.log, h1b.test) #predict classes with default .5 cutoff

#calculate performance with confusion matrix
confusionMatrix(test.pred.class, h1b.test$CASE_STATUS)

#let draw ROC curve of training and test performance of logit model
test.log.roc<- roc(response= h1b.test$CASE_STATUS, predictor=test.pred.prob[[1]]) #assumes postive class Yes is reference level
plot(test.log.roc, legacy.axes=T)
plot(h1B.log.roc, add=T, col="blue")
legend(x=.2, y=.7, legend=c("Test Logit", "Train Logit"), col=c("black", "blue"),lty=1)

#test performance slightly lower than resample
auc(test.log.roc)
auc(h1B.log.roc)

#calculate test confusion matrix using thresholds from resampled data
test.pred.class.newthresh <- factor(ifelse(test.pred.prob[[1]] > h1B.log.Thresh[1], "Yes", "No"))

#recalculate confusion matrix with new cut off predictions
confusionMatrix(test.pred.class.newthresh, h1b.test$CASE_STATUS)


#you have to adjust thresholds when dealing with unbalanced data
#don't ignore cost of FPR or FNR, falsely accusing may be expensive



###BONUS PLOTS to calibrate threshold LIFT AND CALIBRATION
#create a lift chart of logit test probabilities against
test.lift <- lift(h1b.test$CASE_STATUS ~ test.pred.prob[[1]]) #Lift
plot(test.lift)

test.cal <- calibration(h1b.test$CASE_STATUS ~ test.pred.prob[[1]]) #Calibration 
plot(test.cal)


pp.thresh <- glm(h1b.test$CASE_STATUS ~ test.pred.prob[[1]], family="binomial")


predict(pp.thresh, as.data.frame(test.pred.prob[[1]]))
