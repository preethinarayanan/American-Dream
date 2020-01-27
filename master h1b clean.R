setwd("C:/Users/preek/Desktop/SDSU MIS/MIS 620")
library("caret")
library("randomForest")
library("dplyr")
library("glmnet")
library("ROCR")
library("DMwR")
library("ROSE")

h1b <- read.csv('MasterH1BDataset.csv')
h1b <- na.omit(h1b)
str(h1b)

h1b <- subset(h1b, select = -c(CASE_SUBMITTED_DAY, DECISION_DAY, EMPLOYER_NAME, NAICS_CODE, PW_SOURCE, 
                               PW_SOURCE_YEAR, PW_SOURCE_OTHER, WAGE_RATE_OF_PAY_FROM, WAGE_RATE_OF_PAY_TO,
                               WAGE_UNIT_OF_PAY, WORKSITE_POSTAL_CODE))

h1b <- subset(h1b, h1b$VISA_CLASS=="H1B")
h1b <- subset(h1b, h1b$EMPLOYER_COUNTRY=="UNITED STATES OF AMERICA")
h1b <- subset(h1b, select = -c(VISA_CLASS, EMPLOYER_COUNTRY))
h1b <- subset(h1b, h1b$TOTAL_WORKERS=='1')
h1b <- subset(h1b, select = -c(TOTAL_WORKERS))
h1b <- subset(h1b, h1b$CASE_STATUS != "WITHDRAWN")
h1b$CASE_STATUS[h1b$CASE_STATUS == "CERTIFIEDWITHDRAWN"] <- "CERTIFIED"

h1b <- h1b[!(h1b$FULL_TIME_POSITION==""),]
h1b <- h1b[!(h1b$PW_UNIT_OF_PAY==""),]
h1b <- h1b[!(h1b$H.1B_DEPENDENT==""),]
h1b <- h1b[!(h1b$WILLFUL_VIOLATOR==""),]
h1b <- droplevels(h1b)
summary(h1b$CASE_STATUS)
pw_unit_to_yearly <- function(prevailing_wage, pw_unit_of_pay) {
  return(ifelse(pw_unit_of_pay == "Year",
                prevailing_wage,
                ifelse(pw_unit_of_pay == "Hour",
                       2080*prevailing_wage,
                       ifelse(pw_unit_of_pay== "Week",
                              52*prevailing_wage,
                              ifelse(pw_unit_of_pay == "Month",
                                     12*prevailing_wage,
                                     26*prevailing_wage)))))
}


h1b %>%
  filter(!is.na(PW_UNIT_OF_PAY)) %>%
  mutate(PREVAILING_WAGE = as.numeric(PREVAILING_WAGE)) %>%
  mutate(PREVAILING_WAGE = pw_unit_to_yearly(PREVAILING_WAGE, PW_UNIT_OF_PAY)) %>%
  select(everything()) -> h1bconvert

summary(h1bconvert$PREVAILING_WAGE)
boxplot(h1bconvert$PREVAILING_WAGE)

sh1b <- subset(h1bconvert,h1bconvert$PREVAILING_WAGE<10000000)
boxplot(sh1b$PREVAILING_WAGE)

ssh1b <- subset(h1bconvert,h1bconvert$PREVAILING_WAGE<1000000)
boxplot(ssh1b$PREVAILING_WAGE)

h1bconvert <- h1bconvert[!(h1bconvert$PREVAILING_WAGE<10000 | h1bconvert$PREVAILING_WAGE>400000),]
summary(h1bconvert$PREVAILING_WAGE)

h1bconvert <- subset(h1bconvert, select = -c(PW_UNIT_OF_PAY))
h1bclean <- h1bconvert

dummy <- dummyVars( ~., data=h1bclean)
dummy.h1b <- predict(dummy, newdata = h1bclean)
library(DMwR)
#smote <- SMOTE(CASE_STATUS ~ ., data = h1bclean, perc.over = 3000, perc.under = 125)
smote <- SMOTE(CASE_STATUS ~ ., data = h1bclean, perc.over = 2500, perc.under = 105)

table(smote$CASE_STATUS)
rose <- ROSE(CASE_STATUS ~ ., data = h1bclean)$data
table(rose$CASE_STATUS)

h1bsmote <- smote
h1b.bal <- list("Smote"=h1bsmote, "ROSE"=rose)
h1b.resamples = resamples(h1b.bal)

#confusionMatrix(predict(smote_outside,imbal_test), imbal_test$Class)

#compare the performance of all models trained today
rValues <- resamples(list("Smote"=h1bsmote, "ROSE"=rose))
bwplot(rValues, metric="ROC")
bwplot(rValues, metric="Sens") #Sensitvity
bwplot(rValues, metric="Spec")

#plot performance comparisons
bwplot(h1b.resamples, metric="ROC") 
bwplot(h1b.resamples, metric="Sens") #predicting default dependant on threshold
bwplot(h1b.resamples, metric="Spec") 

#calculate ROC curves on resampled data

h1B.glmnet.roc<- roc(response= h1B.glmnet$pred$obs, predictor=h1B.glmnet$pred$Yes)
ctrl <- trainControl(method="cv", number=5,
                     classProbs=TRUE,
                     #function used to measure performance
                     summaryFunction = twoClassSummary, #multiClassSummary for non binary
                     allowParallel =  TRUE)

trainIndex <- createDataPartition(h1bsmote$CASE_STATUS, p = .8, 
                                  list = FALSE, 
                                  times = 1)
head(trainIndex)
h1BTrain <- h1bsmote[ trainIndex,]

#create training set subset (20%)
h1BTest  <- h1bsmote[-trainIndex,]
table(h1BTrain$CASE_STATUS)
table(h1BTest$CASE_STATUS)

library("doParallel")

#Register core backend, using 4 cores
cl <- makeCluster(3)
registerDoParallel(cl)

#list number of workers
getDoParWorkers()


#NAIVE BAYES:
##Naive Bayes
modelLookup("nb") #we have some paramters to tune such as laplace correction
set.seed(192)
library(MLmetrics)
#h1B.nb <- train(CASE_STATUS ~ .-EMPLOYER_NAME - WORKSITE_POSTAL_CODE, data = h1BTrain,
#trControl = ctrl,
#metric = "ROC", #using AUC to find best performing parameters
#method = "nb")
h1B.nb <- train(CASE_STATUS ~ ., data = h1BTrain,
                trControl = ctrl,
                metric = "ROC", #using AUC to find best performing parameters
                method = "nb")
h1B.nb
h1bnb <- save(h1B.nb, file = "naivebayesh1Bfile.rda")
load("naivebayesh1Bfile.rda")
getTrainPerf(h1B.nb)
varImp(h1B.nb)
plot(h1B.nb)
p.nb1<- predict(h1B.nb,h1BTest) 
p.nb<- predict(h1B.nb,h1BTest, type = "prob")
ph1bnb <- save(p.nb, file = "prenbprob.rda")
load("prenbprob.rda")

confusionMatrix(p.nb, h1BTest$CASE_STATUS)
#roc(p.nb, h1BTest$CASE_STATUS,type='prob')
#roc(h1BTest,p.nb)
nb <- save(p.nb, file = "naivebayesh1b.rda")
load("naivebayesh1b.rda")
load("naivebayesh1Bfile.rda")
library(pROC)
#nb.roc <- roc(h1BTest$CASE_STATUS, p.nb, type='prob')
plot(smooth(test.h1bnb.roc), col="green")
#plot(smooth(log.roc), add=T, col="red")
#plot(smooth(lda.roc), add=T, col="blue")
#plot(smooth(logboost.roc), add=T, col="orange")
#legend(x=.33, y=.3, cex=1, legend=c("rpart","logistic","lda","logboost"), col=c("black", "red", "blue", "orange"), lwd=3)
test.h1bnb.roc<- roc(response= h1BTest$CASE_STATUS, predictor=p.nb$DENIED) #assumes postive class Yes is reference level
plot(test.h1bnb.roc, legacy.axes=T)
#plot(h1B.h1bnb.roc, add=T, col="blue")
#legend(x=.2, y=.7, legend=c("Test Logit", "Train Logit"), col=c("black", "blue"),lty=1)

#test performance slightly lower than resample
#auc(test.log.roc)
auc(test.h1bnb.roc)

h1b.model <- list("NB"=h1B.nb, GLMNET=h1B.glmnet)
h1b.resamples = resamples(h1b.model)


#plot performance comparisons
bwplot(h1b.resamples, metric="ROC") 
bwplot(h1b.resamples, metric="Sens") #predicting default dependant on threshold
bwplot(h1b.resamples, metric="Spec") 

#calculate ROC curves on resampled data

h1B.glmnet.roc<- roc(response= h1B.glmnet$pred$obs, predictor=h1B.glmnet$pred$Yes)
h1B.lda.roc<- roc(response= h1B.lda$pred$obs, predictor=h1B.lda$pred$Yes)
h1B.qda.roc<- roc(response= h1B.qda$pred$obs, predictor=h1B.qda$pred$Yes)

##BAGGING - bootstrapping is used to create many training sets and simple models are trained on each and combined
##many small decision trees
library(ipred)
set.seed(192)
modelLookup("treebag")
h1B.bag <- train(CASE_STATUS ~ ., data = h1BTrain,
               trControl = ctrl,
               metric = "ROC", #using AUC to find best performing parameters
               method = "treebag")

h1B.bag
p.bag<- predict(h1B.bag,h1BTest)
confusionMatrix(p.bag,h1BTest$CASE_STATUS)

#cores <- 4 #don't use more than 4 cores
#cl <- makeForkCluster(cores)
#clusterSetRNGStream(cl, 489) #set seed for everymember of cluster
#registerDoParallel(cl)


#Logistic Lasso/Ridge Regression, alpha = ridge vs lasso ratio, lambda=shrinkage amount
h1B.glmnet <- train(CASE_STATUS~., data = h1BTrain, method="glmnet", 
                    metric="ROC", trControl=ctrl)

h1B.glmnet
varImp(h1B.glmnet)
GLMNET <- save(h1B.glmnet, file = "h1B.glmnet.rda")
getTrainPerf(h1B.glmnet)
#varImp(h1B.nb)
plot(h1B.glmnet)
#for confusion matrix use p.nb no type = prob
#for roc use p.glmnet, type="prob..change p.glmnet accordingly
p.glmnet<- predict(h1B.glmnet,h1BTest) 
p.glmnet<- predict(h1B.glmnet,h1BTest, type = "prob")
#confusionMatrix(h1BTest$CASE_STATUS, p.glmnet$CERTIFIED)
confusionMatrix(p.glmnet, h1BTest$CASE_STATUS)
load("h1B.glmnet.rda")
#roc(p.nb, h1BTest$CASE_STATUS,type='prob')
#roc(h1BTest,p.nb)
pglm <- save(p.glmnet, file = "pglmnet.rda")
#load("naivebayesh1b.rda")
library(pROC)
#nb.roc <- roc(h1BTest$CASE_STATUS, p.nb, type='prob')
plot(smooth(test.h1bnb.roc), col="black")
#plot(smooth(log.roc), add=T, col="red")
#plot(smooth(lda.roc), add=T, col="blue")
#plot(smooth(logboost.roc), add=T, col="orange")
#legend(x=.33, y=.3, cex=1, legend=c("rpart","logistic","lda","logboost"), col=c("black", "red", "blue", "orange"), lwd=3)
test.h1bglmnet.roc<- roc(response= h1BTest$CASE_STATUS, predictor=p.glmnet$DENIED) #assumes postive class Yes is reference level
plot(smooth(test.h1bglmnet.roc), col="dark blue")
plot(test.h1bglmnet.roc, legacy.axes=T)
#end cluster after finishing model
stopCluster(cl)

library(ada)
set.seed(192)
#boosted decision trees
#using dummy codeds because this function internally does it and its better to handle it yourself (i.e., less error prone)
modelLookup("ada")
h1B.ada <- train(CASE_STATUS ~ ., data = h1BTrain,
               trControl = ctrl,
               metric = "ROC", #using AUC to find best performing parameters
               method = "ada")
#boosting
#set.seed(44)
#boost.train <- train(Salary ~ ., data=Hitters, 
                    # method="gbm",tuneLength=4,
                    # trControl=ctrl)

h1B.ada
plot(h1B.ada)
p.ada<- predict(h1B.ada, h1BTest)
confusionMatrix(p.ada, h1BTest$CASE_STATUS)


##quadratic distriminant analysis
set.seed(199)
h1B.qda <-  train(CASE_STATUS ~ ., data = h1BTrain, method="qda", metric="ROC", trControl=ctrl)
#h1B.qda <-  train(CASE_STATUS ~ h1b.emplname + h1b.Pos + h1b.PW + CASE_SUBMITTED_MONTH + CASE_SUBMITTED_DAY + CASE_SUBMITTED_YEAR + DECISION_DAY + DECISION_MONTH + DECISION_YEAR + VISA_CLASS +  EMPLOYER_STATE + EMPLOYER_COUNTRY + SOC_NAME + NAICS_CODE + TOTAL_WORKERS + FULL_TIME_POSITION + PREVAILING_WAGE + PW_UNIT_OF_PAY + PW_SOURCE + PW_SOURCE_YEAR + WAGE_RATE_OF_PAY_FROM + WAGE_RATE_OF_PAY_TO + WAGE_UNIT_OF_PAY + H.1B_DEPENDENT + WILLFUL_VIOLATOR + WORKSITE_STATE,data = h1B.train, method="qda", metric="ROC", trControl=ctrl)
h1B.qda
p.qda<- predict(h1B.qda,h1BTest)
confusionMatrix(p.qda,h1BTest$CASE_STATUS)
getTrainPerf(h1B.qda)
varImp(h1B.qda)

#gam splines #not working
set.seed(5627) #set.seed not completely sufficient when multicore
h1B.gam <- train(CASE_STATUS ~ ., data = h1BTrain, 
                 method = "gamSpline",tuneLength=4,
                 metric = "ROC",
                 trControl = ctrl)
h1B.gam
p.rf<- predict(h1B.gam,h1BTest)
confusionMatrix(h1B.gam,h1BTest$CASE_STATUS)

library(randomForest)
set.seed(192)
modelLookup("rf")
#h1B.rf <- train(CASE_STATUS ~ .-EMPLOYER_NAME - WORKSITE_POSTAL_CODE, data = h1BTrain,
#trControl = ctrl,
#metric = "ROC", #using AUC to find best performing parameters
#tuneLength=9,
#method = c("rf") )
h1B.rf <- train(CASE_STATUS ~ ., data = h1BTrain,
                trControl = ctrl,
                metric = "ROC", #using AUC to find best performing parameters
                tuneLength=9,
                method = c("rf") )
h1B.rf
h1brf <- save(h1B.rf, file = "h1brf.rda")
p.rf<- predict(h1B.rf,h1BTest)
confusionMatrix(h1B.rf,h1BTest$CASE_STATUS)

pca
library(AppliedPredictiveModeling)
#CASE_STATUS <- factor(c("CERTIFIED","WITHDRAWN"))

pp_hpc1 <- preProcess(h1bdrop[, ], 
                      method = c("center", "scale", "BoxCox"))
pp_hpc1 #can see results of processing

#now we juse need to apply the processing model to the data
#transformed <- predict(pp_hpc, newdata = h1B[, -8,-26])
#head(transformed)

transformed1 <- predict(pp_hpc1, newdata = h1bdrop[, ])
head(transformed1)


#numpending very low variance
mean(h1bdrop$NumPending == 0) #76%=0

#feature reduction using PCA

pp_no_pca1 <- preProcess(h1bdrop[, ], method = c("pca"))
pp_no_pca1

head(predict(pp_no_pca1, newdata = h1bdrop[ , ]))
finaldata <- (predict(pp_no_pca1, newdata = h1bdrop[ , ]))
