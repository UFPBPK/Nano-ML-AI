## Loading the H2O and other deep learning packages
library(keras)
library(mlbench)
library(dplyr)
library(magrittr)
library(neuralnet)
library(h2o)
library(caret)
library(ggplot2)

## Start the H2O package
h2o.init(nthreads=-1, max_mem_size="2G") # allow it to use all CPU cores and up to 2GB of memory:
h2o.removeAll() ## clean cluster - just in case the cluster was already running

# Loading data
data <- read.csv(file = 'data.csv')    ## Read the data set
data <- data%>%select(Type:ZP, DE_Max) ## Select feature and response variables
data <- data%>%filter(!is.na(ZP))      ## Filter miss data
data <- data%>%filter(DE_Max < 50)     ## Eliminate the outlier

## Check data structure
str(data)

##------------------------------------------------------------------------------
# Remove the outliers in response variable (Option)
# Q   <- quantile(data$DE_Max, probs = c(.05, .95), na.rm = FALSE) # estimate the quantile
# iqr <- IQR(data$DE_Max)   ## Estimate the IQR
# up  <- Q[2]#+1.5*iqr      ## Upper Range
# low <- Q[1]#-1.5*iqr      ## Lower Range
# eliminated<- subset(data, data$DE_Max > Q[1])# & data$DE_Max < Q[2])
# ggstatsplot::ggbetweenstats(eliminated, Type, DE_Max, outlier.tagging = TRUE)
##------------------------------------------------------------------------------

## One-Hot encoding: Function to encode the categorical variables as factors
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

# Encode the categorical variables as factors
data_r <- data %>% tofactor 
str(data_r)

## Scale data: Normalized value = (value - mean)/sd
scaled      <-caret::preProcess(data_r[,-dim(data_r)[2]], method = c("center"))
transformed <-predict(scaled, newdata = data_r)

## One-hot encoding the categorical variables
dummies     <-dummyVars(DE_Max~., data = transformed)
mm          <-predict(dummies, newdata = transformed)

## "as.h2o": Create H2O Frame
MM<-as.h2o(mm %>% as.data.frame %>% mutate(Y=transformed[,"DE_Max"]))


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
print(shortlistedVars)

# Create training (80%) and test (20%) sets 
# Use set.seed = 123 for reproducibility
splits <- h2o.splitFrame(MM, c(0.8), seed=123)
splits_lable <- h2o.splitFrame(as.h2o(data_r), c(0.8), seed=123)
train  <- h2o.assign(splits[[1]], "train.hex") # 80%
valid  <- h2o.assign(splits[[2]], "valid.hex") # 20%

## Define the response and predictors variables
response <- "Y"
predictors <- setdiff(shortlistedVars, response)
predictors

##------------------------------------------------------------------------------
## Model tuning manually ##--Final model # "3" indicates the third version of the model
DL_DE_MAX_3 <- h2o.deeplearning(
  model_id             = "DL_DE_MAX_tuned",
  training_frame       = train[,c(predictors,response)], ## Training data set: used for model training
  validation_frame     = valid[,c(predictors,response)], ## Validation data set: used for scoring and early stopping if validation result is not acceptable
  x                    = predictors,     ## Predictor variables
  y                    = response,       ## Response variables
  hidden               = c(512,256,128), ## Set up the layers of neural network
  epochs               = 5000000,        ## Iterations
  nfolds               = 5,              ## 5-fold cross validation 
  fold_assignment      = "Modulo",       ## Cross-validation fold assignment scheme 
  loss                 = "Automatic",    ## Loss function
  #distribution         = "AUTO",        ## Distribution function
  score_validation_samples = 100000,     ## downsample validation set for faster scoring
  score_duty_cycle     = 0.025,          ## don't score more than 2.5% of the wall time
  stopping_rounds      = 200,            ## used for early stopping
  stopping_tolerance   = 0.0001,         ## Specify the relative tolerance for the metric-based stopping to stop training if the improvement is less than this value
  stopping_metric      = "AUTO",         ## Specify the metric to use for early stopping
  adaptive_rate        = F,              ## manually tune learning rate
  activation           = "RectifierWithDropout",## Specify the activation function
  #input_dropout_ratio   = 0.1,          ## Specify the input layer dropout ratio to improve generalization
  hidden_dropout_ratios = c(0.5,0.2,0.1),## Specify the nodes of hidden layers 
  rate                 = 0.005,          ## Specify the learning rate
  #rate_annealing       = 1e-05,         ## Specify the rate annealing value.
  #rho                  = 0.99,          ## Specify the adaptive learning rate time decay factor
  #epsilon              = 1e-8,          ## Specify the adaptive learning rate time smoothing factor to avoid dividing by zero
  #momentum_start       = 0.2,           ## Manually tuned momentum
  #momentum_stable      = 0.99,          ## Specify the final momentum after the ramp is over
  #momentum_ramp        = 1e6,           ## Specify the number of training samples for which the momentum increases
  l1                   = 1e-6,#1e-6,     ## add some L1/L2 regularization
  l2                   = 1e-6,#1e-6,     ## add some L1/L2 regularization
  max_w2               = 10,             ## Stable the ReLu activation function
  sparse               = T,              ## Specify whether to enable sparse data handling
  variable_importances = T,              ## Specify whether to compute variable importance
  reproducible         = T,              ## Force reproducibility (Only for small data)
  seed                 = 123             ## not enabled by default
)

## Summary
summary(DL_DE_MAX_3)
plot(DL_DE_MAX_3)

## Save the model
model_path_DE_MAX <- h2o.saveModel(object = DL_DE_MAX_3, 
                                   path = getwd(), 
                                   force = TRUE)


## 
print(model_path_DE_MAX)


##----------------------------------------
## Prediction

pred_train <- as.data.frame(h2o.predict(DL_DE_MAX_3, train[,c(predictors,response)]))%>%
  mutate(dtype="train", MAT=as.vector(splits_lable[[1]]$MAT))%>%
  rename(Pred = predict)

pred_test  <- as.data.frame(h2o.predict(DL_DE_MAX_3, valid[,c(predictors,response)]))%>%
  mutate(dtype="test", MAT=as.vector(splits_lable[[2]]$MAT))%>%
  rename(Pred = predict)


pred  <- rbind.data.frame(pred_train, pred_test)

obs   <- rbind.data.frame(as.matrix(train[,response]),
                          as.matrix(valid[,response]))%>%rename(Obs = Y)


PlotDat <- cbind.data.frame(obs,pred)

## Rename levels
PlotDat$dtype<-forcats::fct_recode(PlotDat$dtype, Test_set = "test", Training_set = "train")

## Reorder levels
PlotDat$dtype<-factor(PlotDat$dtype,levels(PlotDat$dtype)[c(2,1)])


p<-ggplot(PlotDat, aes(x = Obs, y=Pred, shape = factor(MAT), color = factor(dtype))) +
  geom_point(size = 4) +
  scale_shape_manual(values=13:(13+9),labels = levels(factor(data$MAT)))+
  scale_colour_grey(start = 0.7, end = 0.2,labels = levels(factor(PlotDat$dtype)))+
  guides(colour = guide_legend(title="Data type",override.aes = list(shape = 15)),
         shape =  guide_legend(title="NPs"))+

    geom_abline (intercept = 0,
               slope     = 1,
               color     ="black",size = 1)+
  scale_x_continuous(limits = c(0, 30))+
  scale_y_continuous(limits = c(0, 30))

p

windowsFonts(Times=windowsFont("Times New Roman"))

p1 <- p +
  theme (
    legend.key = element_rect(fill = "white", color = NA),
    legend.text = element_text(size=15),
    plot.background         = element_rect (fill="White"),
    text                    = element_text (family = "Times"),   # text front (Time new roman)
    panel.border            = element_rect (colour = "black", fill=NA, size=2),
    panel.background        = element_rect (fill = NA),
    panel.grid.major        = element_blank(),
    panel.grid.minor        = element_blank(),
    axis.text               = element_text (size   = 20, colour = "black", face = "bold"),    # tick labels along axes
    axis.title              = element_text (size   = 20, colour = "black", face = "bold"),   # label of axes
  ) +
  labs (x = "", y = "")

p1


## Save the figure
# ggsave("Fig.3a.tiff",scale = 1,
#        plot = p1,
#        width = 25, height = 18, units = "cm", dpi=320)
# 
