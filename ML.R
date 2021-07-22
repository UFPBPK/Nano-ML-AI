## Loading R packages
library(caret)
library(tidyverse)
library(doParallel)
library(Boruta)
library(glmnet)
library(randomForest)
library(gbm)
library(e1071)
library(kernlab)
library(LiblineaR)

## Start the H2O
h2o.init(nthreads=-1, max_mem_size="2G") # allow it to use all CPU cores and up to 2GB of memory:
h2o.removeAll() ## clean slate - just in case the cluster was already running

## Create a function to conduct the data preprocessing

Dat_preprocessing <- function(Variable = "DE_Max") {
## Loading data
data <- read.csv(file = 'data.csv') ## Read the dataset
data <- data%>%select(Type:ZP, Variable) ## Select feature and response variables
data <- data%>%filter(!is.na(ZP)) ## Eliminate the outlier

## Check data structure
#str(data)

##------------------------------------------------------------------------------
# Optional equation: Remove the outliers in response variable
# Q   <- quantile(data$DE_24, probs = c(.025, .975), na.rm = FALSE) # estimate the quantile
# iqr <- IQR(data$DE_Max)   ## Estimate the IQR
# up  <- Q[2]#+1.5*iqr      ## Upper Range
# low <- Q[1]#-1.5*iqr      ## Lower Range
# eliminated<- subset(data, data$DE_24 > Q[1])# & data$DE_Max < Q[2])
# ggstatsplot::ggbetweenstats(eliminated, Type, DE_24, outlier.tagging = TRUE)
# data <- eliminated
#------------------------------------------------------------------------------


## Function to encode the categorical variables
tofactor<- function (x) {
  for (i in 1:ncol(x)){
    if (is.character(x[,i]) == TRUE){
      x[,i] = factor(x[,i],
                     levels = levels(factor(x[,i])),
                     labels = seq(1:length(levels(factor(x[,i])))))
    }
  }
  return(x)
}

## Encoding the categorical variables 
data_r <- data %>% tofactor 
#str(data_r)

## Scale data: Normalized value = (value - mean)/sd
scaled      <-caret::preProcess(data_r[,-dim(data_r)[2]], method = c("center"))
transformed <-predict(scaled, newdata = data_r)
colnames(transformed)[9]<-"Y"

## One-hot encoding the categorical variables
dummies     <- dummyVars(Y~., data = transformed)
mm          <- predict(dummies, newdata = transformed)

## "as.h2o": Create h2OFrame for use in h2o package
MM<-as.h2o(mm %>% as.data.frame %>% mutate(Y=transformed[,"Y"]))


## Feature selection: Step-wise regression (optional)
# base.mod <- lm(Y~1, data = as.data.frame(MM))
# all.mod  <- lm(Y~., data = as.data.frame(MM)) # full model with all predictors
# stepMod <- step(base.mod,
#                 scope = list(lower = base.mod, upper = all.mod),
#                 direction = "both", trace = 0, steps = 100)  # perform step-wise algorithm
# 
# shortlistedVars <- names(unlist(stepMod[[1]])) # get the shortlisted variable.
# shortlistedVars <- shortlistedVars[!shortlistedVars %in% "(Intercept)"]  # remove intercept
# print(shortlistedVars)

## Feature selection: Zero and near-zero variance features
feature_variance <- caret::nearZeroVar(MM, saveMetrics = TRUE)
shortlistedVars  <- colnames(MM[, feature_variance$nzv == 'FALSE'])
#print(shortlistedVars)

# Create training (80%) and test (20%) sets 
# Use set.seed = 123 for reproducibility
splits <- h2o.splitFrame(MM, c(0.8), seed=123)
splits_lable <- h2o.splitFrame(as.h2o(data_r), c(0.8), seed=123)
train  <- h2o.assign(splits[[1]], "train.hex") # 80%
valid  <- h2o.assign(splits[[2]], "valid.hex") # 20%

## Define the response and predictors variables
response <- "Y"
predictors <- setdiff(shortlistedVars, response)

## Change the data frame
Dat_train <- as.data.frame(train[,c(predictors,response)])
X_test    <- as.data.frame(valid[,c(predictors,response)])%>%select(-Y)
Y_test    <- as.data.frame(valid[,c(predictors,response)])%>%select(Y)


return (list(Dat_train, X_test,Y_test))
}



## Create the ML function
ML <- function (data, cv = 5, rep = 5, method) {
  # Set the seed for reproducibility
  set.seed(123)
  
    ## Create control function for training  
    ctrl <- trainControl(method  = 'repeatedcv', 
                             number  = cv, # 5-fold cv  
                             repeats = rep,
                             search  ='grid') # search method is grid.)
    model <- train(Y ~ .,
                   data = as.data.frame(data), 
                   method = method,
                   trControl = ctrl,
                   metric = "RMSE")        
        
return(model)
} 

## 
Dat_DE_Max  <- Dat_preprocessing(Variable = "DE_Max")
Dat_DE_TMax <- Dat_preprocessing(Variable = "DE_Tmax")
Dat_DE_24   <- Dat_preprocessing(Variable = "DE_24")
Dat_DE_168  <- Dat_preprocessing(Variable = "DE_168")


## Set the seed for reproducibility
set.seed(123)

## Linear
lmFit_DE_Max <-ML (data=Dat_DE_Max[1], method = 'glmnet') 
lmFit_DE_TMax<-ML (data=Dat_DE_TMax[1], method = 'glmnet') 
lmFit_DE_24  <-ML (data=Dat_DE_24[1], method = 'glmnet') 
lmFit_DE_168 <-ML (data=Dat_DE_168[1], method = 'glmnet') 

## KNN
knnFit_DE_Max <-ML (data=Dat_DE_Max[1], method = 'knn') 
knnFit_DE_TMax<-ML (data=Dat_DE_TMax[1], method = 'knn') 
knnFit_DE_24  <-ML (data=Dat_DE_24[1], method = 'knn') 
knnFit_DE_168 <-ML (data=Dat_DE_168[1], method = 'knn') 

## RF
rfFit_DE_Max <-ML (data=Dat_DE_Max[1], method = 'rf') 
rfFit_DE_TMax<-ML (data=Dat_DE_TMax[1], method = 'rf') 
rfFit_DE_24  <-ML (data=Dat_DE_24[1], method = 'rf') 
rfFit_DE_168 <-ML (data=Dat_DE_168[1], method = 'rf') 

## Bag
bagFit_DE_Max <-ML (data=Dat_DE_Max[1], method = 'treebag') 
bagFit_DE_TMax<-ML (data=Dat_DE_TMax[1], method = 'treebag') 
bagFit_DE_24  <-ML (data=Dat_DE_24[1], method = 'treebag') 
bagFit_DE_168 <-ML (data=Dat_DE_168[1], method = 'treebag') 

## GBM
gbmFit_DE_Max <-ML (data=Dat_DE_Max[1], method = 'gbm') 
gbmFit_DE_TMax<-ML (data=Dat_DE_TMax[1], method = 'gbm') 
gbmFit_DE_24  <-ML (data=Dat_DE_24[1], method = 'gbm') 
gbmFit_DE_168 <-ML (data=Dat_DE_168[1], method = 'gbm') 

## SVM-1
svmFit1_DE_Max <-ML (data=Dat_DE_Max[1], method = 'svmLinear') 
svmFit1_DE_TMax<-ML (data=Dat_DE_TMax[1], method = 'svmLinear') 
svmFit1_DE_24  <-ML (data=Dat_DE_24[1], method = 'svmLinear') 
svmFit1_DE_168 <-ML (data=Dat_DE_168[1], method = 'svmLinear') 

### LS-SVM
svmFit2_DE_Max <-ML (data=Dat_DE_Max[1], method = 'svmLinear2') 
svmFit2_DE_TMax<-ML (data=Dat_DE_TMax[1], method = 'svmLinear2') 
svmFit2_DE_24  <-ML (data=Dat_DE_24[1], method = 'svmLinear2') 
svmFit2_DE_168 <-ML (data=Dat_DE_168[1], method = 'svmLinear2') 

## L2-SVM-3
svmFit3_DE_Max <-ML (data=Dat_DE_Max[1], method = 'svmLinear3') 
svmFit3_DE_TMax<-ML (data=Dat_DE_TMax[1], method = 'svmLinear3') 
svmFit3_DE_24 <-ML (data=Dat_DE_24[1], method = 'svmLinear3') 
svmFit3_DE_168 <-ML (data=Dat_DE_168[1], method = 'svmLinear3') 


## collect re-samples
results_DE_Max <- resamples(list(LM   = lmFit_DE_Max, 
                            KNN  = knnFit_DE_Max,
                            RF   = rfFit_DE_Max,
                            BAG  = bagFit_DE_Max,
                            GBM  = gbmFit_DE_Max,
                            SVM1 = svmFit1_DE_Max, 
                            SVM2 = svmFit2_DE_Max, 
                            SVM3 = svmFit3_DE_Max))

##
results_DE_TMax <- resamples(list(LM   = lmFit_DE_TMax, 
                             KNN  = knnFit_DE_TMax,
                             RF   = rfFit_DE_TMax,
                             BAG  = bagFit_DE_TMax,
                             GBM  = gbmFit_DE_TMax,
                             SVM1 = svmFit1_DE_TMax, 
                             SVM2 = svmFit2_DE_TMax, 
                             SVM3 = svmFit3_DE_TMax))

## 
results_DE_24 <- resamples(list(LM   = lmFit_DE_24, 
                                 KNN  = knnFit_DE_24,
                                 RF   = rfFit_DE_24,
                                 BAG  = bagFit_DE_24,
                                 GBM  = gbmFit_DE_24,
                                 SVM1 = svmFit1_DE_24, 
                                 SVM2 = svmFit2_DE_24, 
                                 SVM3 = svmFit3_DE_24))

##
results_DE_168<- resamples(list(LM   = lmFit_DE_168, 
                                KNN  = knnFit_DE_168,
                                RF   = rfFit_DE_168,
                                BAG  = bagFit_DE_168,
                                GBM  = gbmFit_DE_168,
                                SVM1 = svmFit1_DE_168, 
                                SVM2 = svmFit2_DE_168, 
                                SVM3 = svmFit3_DE_168))

## Summarize differences between models
library(rstatix)

summary_DE_MAX<-results_DE_Max$values%>%
  rstatix::get_summary_stats(show = c("mean", "sd"))

summary_DE_TMAX<-results_DE_TMax$values%>%
  rstatix::get_summary_stats(show = c("mean", "sd"))

summary_DE_24<-results_DE_24$values%>%
  rstatix::get_summary_stats(show = c("mean", "sd"))

summary_DE_168<-results_DE_168$values%>%
  rstatix::get_summary_stats(show = c("mean", "sd"))


## Box and whisker plots to compare models
# scales <- list(x=list(relation="free"), y=list(relation="free"))
# bwplot(results_DE_Max, scales=scales)
# bwplot(results_DE_TMax, scales=scales)
# bwplot(results_DE_24, scales=scales)
# bwplot(results_DE_168, scales=scales)

## Model Testing

## DE_MAX
lm_DE_Max_test     <- lmFit_DE_Max%>%predict(newdata = Dat_DE_Max[2])
knn_DE_Max_test    <- knnFit_DE_Max%>%predict(newdata = Dat_DE_Max[2])
rfFit_DE_Max_test  <- rfFit_DE_Max%>%predict(newdata = Dat_DE_Max[2])
bagFit_DE_Max_test <- bagFit_DE_Max%>%predict(newdata = Dat_DE_Max[2])
gbmFit_DE_Max_test <- gbmFit_DE_Max%>%predict(newdata = Dat_DE_Max[2])
svmFit1_DE_Max_test<- svmFit1_DE_Max%>%predict(newdata = Dat_DE_Max[2])
svmFit2_DE_Max_test<- svmFit2_DE_Max%>%predict(newdata = Dat_DE_Max[2])
svmFit3_DE_Max_test<- svmFit3_DE_Max%>%predict(newdata = Dat_DE_Max[2])

## DE_TMAX
lm_DE_TMax_test     <- lmFit_DE_TMax%>%predict(newdata = Dat_DE_TMax[2])
knn_DE_TMax_test    <- knnFit_DE_TMax%>%predict(newdata = Dat_DE_TMax[2])
rfFit_DE_TMax_test  <- rfFit_DE_TMax%>%predict(newdata = Dat_DE_TMax[2])
bagFit_DE_TMax_test <- bagFit_DE_TMax%>%predict(newdata = Dat_DE_TMax[2])
gbmFit_DE_TMax_test <- gbmFit_DE_TMax%>%predict(newdata = Dat_DE_TMax[2])
svmFit1_DE_TMax_test<- svmFit1_DE_TMax%>%predict(newdata = Dat_DE_TMax[2])
svmFit2_DE_TMax_test<- svmFit2_DE_TMax%>%predict(newdata = Dat_DE_TMax[2])
svmFit3_DE_TMax_test<- svmFit3_DE_TMax%>%predict(newdata = Dat_DE_TMax[2])

## DE_24
lm_DE_24_test     <- lmFit_DE_TMax%>%predict(newdata = Dat_DE_24[2])
knn_DE_24_test    <- knnFit_DE_TMax%>%predict(newdata = Dat_DE_24[2])
rfFit_DE_24_test  <- rfFit_DE_TMax%>%predict(newdata = Dat_DE_24[2])
bagFit_DE_24_test <- bagFit_DE_TMax%>%predict(newdata = Dat_DE_24[2])
gbmFit_DE_24_test <- gbmFit_DE_TMax%>%predict(newdata = Dat_DE_24[2])
svmFit1_DE_24_test<- svmFit1_DE_TMax%>%predict(newdata = Dat_DE_24[2])
svmFit2_DE_24_test<- svmFit2_DE_TMax%>%predict(newdata = Dat_DE_24[2])
svmFit3_DE_24_test<- svmFit3_DE_TMax%>%predict(newdata = Dat_DE_24[2])

## DE_168
lm_DE_168_test     <- lmFit_DE_168%>%predict(newdata = Dat_DE_168[2])
knn_DE_168_test    <- knnFit_DE_168%>%predict(newdata = Dat_DE_168[2])
rfFit_DE_168_test  <- rfFit_DE_168%>%predict(newdata = Dat_DE_168[2])
bagFit_DE_168_test <- bagFit_DE_168%>%predict(newdata = Dat_DE_168[2])
gbmFit_DE_168_test <- gbmFit_DE_168%>%predict(newdata = Dat_DE_168[2])
svmFit1_DE_168_test<- svmFit1_DE_168%>%predict(newdata = Dat_DE_168[2])
svmFit2_DE_168_test<- svmFit2_DE_168%>%predict(newdata = Dat_DE_168[2])
svmFit3_DE_168_test<- svmFit3_DE_168%>%predict(newdata = Dat_DE_168[2])


## Prediction 
pred_DE_Max_test<-cbind.data.frame(
                  lm_DE_Max_test,knn_DE_Max_test,rfFit_DE_Max_test, 
                  bagFit_DE_Max_test,gbmFit_DE_Max_test,svmFit1_DE_Max_test, 
                  svmFit2_DE_Max_test,svmFit3_DE_Max_test)

pred_DE_TMax_test<-cbind.data.frame(
  lm_DE_TMax_test,knn_DE_TMax_test,rfFit_DE_TMax_test, 
  bagFit_DE_TMax_test,gbmFit_DE_TMax_test,svmFit1_DE_TMax_test, 
  svmFit2_DE_TMax_test,svmFit3_DE_TMax_test)

pred_DE_24_test<-cbind.data.frame(
  lm_DE_24_test,knn_DE_24_test,rfFit_DE_24_test, 
  bagFit_DE_24_test,gbmFit_DE_24_test,svmFit1_DE_24_test, 
  svmFit2_DE_24_test,svmFit3_DE_24_test)

pred_DE_168_test<-cbind.data.frame(
  lm_DE_168_test,knn_DE_168_test,rfFit_DE_168_test, 
  bagFit_DE_168_test,gbmFit_DE_168_test,svmFit1_DE_168_test, 
  svmFit2_DE_168_test,svmFit3_DE_168_test)


post_results_DE_Max <-apply(pred_DE_Max_test, 2, postResample, obs = as.matrix(Dat_DE_Max[[3]]))
post_results_DE_TMax <-apply(pred_DE_TMax_test, 2, postResample, obs = as.matrix(Dat_DE_TMax[[3]]))
post_results_DE_24 <-apply(pred_DE_24_test, 2, postResample, obs = as.matrix(Dat_DE_24[[3]]))
post_results_DE_168 <-apply(pred_DE_168_test, 2, postResample, obs = as.matrix(Dat_DE_168[[3]]))

