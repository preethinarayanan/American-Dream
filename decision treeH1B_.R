h1b <- read.csv("MasterH1BDataset.csv")
library(caret)
#worksite postal code removed, employer name converted to numeric
h1bclean <- na.omit(h1b)
h1bclean$EMPLOYER_NAME <- as.numeric(h1bclean$EMPLOYER_NAME)
h1bclean <- subset(h1bclean, select = -c(WORKSITE_POSTAL_CODE))
trainIndex <- createDataPartition(h1bclean$CASE_STATUS, p=.7, list=F)   
h1b.train <- h1bclean[trainIndex,]

ctrl <- trainControl(method="cv", number=3, classProbs=TRUE)
h1b.rpart <- train(CASE_STATUS ~ ., data=h1b.train, trControl = ctrl, metric="ROC", method="rpart")
rpart.plot(h1b.rpart$finalModel)
