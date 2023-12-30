#(------------- PART-1: Supervised Learning ----------------------------)

#Load the data
load("class_data.RData")

#Convert dependent variable to categorical format, has two classes (0 and 1)
y <- as.factor(y)

#Set random seed for reproducibility
set.seed(22)

# standardize x values
x_scaled <- scale(x)

#check if any missing values
sum(is.na(x))
print("no missing values")


library(caret)

#Calculate vector showing which variables have near-zero variance
nz_vector <- nearZeroVar(x)

#Obtain the names of the near-zero variance variables
names_near_zero_var <- names(x)[nz_vector]

#Initializing Outer fold Ids
CV_fold = rep(0, dim(x)[1])

for(i in seq(1,length(CV_fold), by = 5)){
  CV_fold[i]<-1
  CV_fold[i+1] <- CV_fold[i] + 1
  CV_fold[i+2] <- CV_fold[i] + 2
  CV_fold[i+3] <- CV_fold[i] + 3
  CV_fold[i+4] <- CV_fold[i] + 4
}

foldsOut = sample(CV_fold)


#-----------------------------Different Models---------------------------------

#
#----------------------------- 1. Logistic Regression--------------------------
#

#Import required libraries
library(mlbench)
library(glmnet)
library(pROC)

#make correlation matrix for finding relationships between variables
CorMtrx <- cor(x)
#find attributes that have high correlation
varsHighCorrelated <- findCorrelation(CorMtrx, cutoff = 0.80)

#Removing the variables that have high correlation
x_logReg <- x[,-c(varsHighCorrelated)]

# No of folds for K-fold cross validation
K = 5

#initializing variables to store the classification error for LR
logRegFoldError = rep(0,K)
auc_logReg  =  rep(0,K)

for(j in 1:K){

#Splitting data into train and val set
x_trainSet <- x_logReg[foldsOut != j,]
y_trainSet <- y[foldsOut != j]

x_valSet <- x_logReg[foldsOut == j,]
y_valSet <- y[foldsOut == j]

#variable selection procedure based on significance of logistic reg. coeff.
f = 1
indx =  c()
for(a in 1:length(x_trainSet)){
xCurrSplit  =  x_trainSet[a]
names(xCurrSplit) = "X"
glm.fit  =  glm(y_trainSet~X,data = xCurrSplit,family = binomial)
if(summary(glm.fit)$coefficients[2,4]<0.01){
  indx[f]  =  a
  f = f+1
}}

x_trainSet <- x_trainSet[,indx]
x_valSet <- x_valSet[,indx]

train  =  data.frame(x_trainSet,y_trainSet)
val = data.frame(x_valSet,y_valSet)

#Fit logistic regression model to train set
glm.fit = glm(y_trainSet~.,data = train,family = binomial)

#Predict on val set & calculate cross-validation fold error
glm.probs = predict(glm.fit,newdata = val,type = 'response')
glm.pred = ifelse(glm.probs>0.5,1,0)
logRegFoldError[j]  = mean(glm.pred != y_valSet)
auc_logReg[j]<- auc(y_valSet,as.numeric(glm.pred))
}

cvError_mean_logReg  =  mean(logRegFoldError)
cvError_mean_logReg

logReg_meanAUC  = mean(auc_logReg)
logReg_meanAUC

#
#------------------- 2. Linear Discriminant Analysis-------------#
#

#Import required libraries
library(MASS)

# No of folds for K-fold cross validation
K = 5

#initializing variables to store the classification error for LDA
LDAFoldError=rep(0,K)
aucFold_LDA = rep(0,K)

for(j in 1:K){
  
  #Splitting into training and validation set
  x_trainSet <- x_logReg[foldsOut!=j,]
  y_trainSet <- y[foldsOut!=j]
  
  x_valSet <- x_logReg[foldsOut==j,]
  y_valSet <- y[foldsOut==j]
  
  #performing a variable selection procedure based on the significance of the logistic regression coefficients
  f=1
  indx= c()
  for(a in 1:length(x_trainSet)){
    xCurrSplit=x_trainSet[a]
    names(xCurrSplit)="X"
    glm.fit = glm(y_trainSet~X,data=xCurrSplit,family=binomial)
    if(summary(glm.fit)$coefficients[2,4]<0.01){
      indx[f] = a
      f=f+1
    }}
  
  x_trainSet <- x_trainSet[,indx]
  x_valSet <- x_valSet[,indx]
  
  train=data.frame(x_trainSet,y_trainSet)
  val=data.frame(x_valSet,y_valSet)
  #Fit LDA model
  LDAFit=lda(y_trainSet~.,data=train)
  
  #Predict on val set and calculate cross-val fold error
  LDAPredVal=predict(LDAFit,newdata=val)
  
  LDAFoldError[j]=mean(LDAPredVal$class!=y_valSet)
  aucFold_LDA[j] <- auc(y_valSet,as.numeric(LDAPredVal$class))
}

cvError_mean_LDA = mean(LDAFoldError)
cvError_mean_LDA

LDA_meanAUC <-mean(aucFold_LDA)
LDA_meanAUC

#
#------------------------ 3. Support Vector Machine----------------------#
#

#Import required libraries
library(e1071)
library(readxl)
library(Boruta)
library(ROCR)
library(Boruta)
library(pROC)
library(ROCit)

# No of folds for K-fold cross validation
K = 5

#initializing variables to store the classification error for SVM
SVMFoldError = rep(0,K)
auc_SVM  = rep(0,K)
costBest  =  rep(0,K)
gammaBest  =  rep(0,K)

#K-fold cross validation loop
for(i in 1:K){
#Splitting data into train and val set
trainSet  <- x[foldsOut != i,]
y_trainSet <- y[foldsOut != i]

valSet <- x[foldsOut == i,]
y_valSet <- y[foldsOut == i]

#Boruta is used to identify relevant features in a dataset
selectedFeatures <- Boruta(trainSet ,y_trainSet)

#vector of the names of the features that have been identified as significant
significantFeatures <- getSelectedAttributes(selectedFeatures, withTentative  =  TRUE)

#Using selected features for tuning & model fitting & scaling
x_trainSet<-scale(trainSet [,significantFeatures])
x_valSet<-scale(valSet[,significantFeatures])

#Using tuning function
fitSVM <- tune.svm(x = x_trainSet, y = y_trainSet, kernel = 'radial',gamma = c(0.01,0.1,1,10,20),cost = c(0.01,0.1,1,10,100))

#Storing best gamma and cost values obtained after tuning
costBest[i] <- fitSVM$best.parameters$cost
gammaBest[i] <- fitSVM$best.parameters$gamma

singleDF  =  data.frame(x_trainSet,y_trainSet)

#Fit svm model on training data with tuned parameters
modelSVM = svm(y_trainSet ~. , kernel  =  'radial', cost  = fitSVM$best.parameters$cost, gamma  =  fitSVM$best.parameters$gamma, data  =  singleDF)

#predictions for validation set
predictionSVM  =  predict(modelSVM,x_valSet)

#validation error and auc score for each fold
SVMFoldError[i] <- mean(predictionSVM != y_valSet)
auc_SVM[i] <-auc(y_valSet, as.numeric(predictionSVM))

#Plotting ROC curve for each fold
ROCPerFoldSVM <- rocit(score = as.numeric(predictionSVM),class = y_valSet)
plot(ROCPerFoldSVM)
}
costBest
gammaBest

#Taking a mean of cv error across K folds
cvError_mean_SVM <-mean(SVMFoldError)
cvError_mean_SVM

#Checking mean auc score of svm model
SVM_meanAUC <-mean(auc_SVM)
SVM_meanAUC

#
#----------------------- 4. KNN Model with given data-----------------#
#

#Import required libraries
library(class)
library(Boruta)
library(pROC)
library(ROCit)
library(ROCR)

#initializing a variable to store the classification error for KNN
KNNError  =  data.frame()
auc_KNN  =  data.frame()

# No of folds for K-fold cross validation
K = 5

#K fold cross validation loop
for(i in 1:K){
  #Splitting data into train set and validation sets
  trainSet <- x_scaled[foldsOut != i,]
  y_trainSet <- y[foldsOut != i]

  valSet <- x_scaled[foldsOut == i,]
  y_valSet <- y[foldsOut == i]
  
  #Boruta is used to identify relevant features in a dataset
  selectedFeatures <- Boruta(trainSet,y_trainSet)
  
  #vector of the names of the features that have been identified as significant
  significantFeatures <- getSelectedAttributes(selectedFeatures, withTentative  =  TRUE)
  
  #Using selected features for tuning & model fitting & scaling in KNN
  x_trainSet<-trainSet[,significantFeatures]
  x_valSet<-valSet[,significantFeatures]
  
  #Performing an iteration over the k nearest neighbors for every fold
  nearest_neighbours = dim(x_trainSet)[1]
  for(j in 1:nearest_neighbours){
    modelKNN = knn(train = x_trainSet, cl = y_trainSet, test = x_valSet, k = j, prob = TRUE)
    KNNError[i,j] = mean(modelKNN != y_valSet)
    auc_KNN[i,j] <- roc(y_valSet,attributes(modelKNN)$prob)$auc
  }
}
#Collecting the average cross-validation error and AUC score for every k value
cvError_mean_KNN  =  apply(KNNError,2,mean)
knn_which.neighbour.min = which.min(cvError_mean_KNN)

KNN_meanAUC  =  apply(auc_KNN,2,mean)
auc_k  = KNN_meanAUC[knn_which.neighbour.min]

#test error is equal to minimum cross validation error
test_errorKNN <-cvError_mean_KNN[knn_which.neighbour.min]

#Print minimum cross-validation error and maximum AUC score
#for the value of k that matches the number displayed in the variable name
print(cvError_mean_KNN[knn_which.neighbour.min])
print(auc_k)

#Relationship b/w the no of neighbors and variation of cross-validation error
plot(1:320,cvError_mean_KNN,xlab  = "Number of nearest Neighbours", ylab  =  "CV Error Mean")

#
#------------------------- 5. Random Forest ------------------------#
#
#Initializing inner fold Ids
foldsIn =rep(0,320)

for(i in seq(1,length(foldsIn),by=4)){
  foldsIn[i]<-1
  foldsIn[i+1] <- foldsIn[i] + 1
  foldsIn[i+2] <- foldsIn[i] + 2
  foldsIn[i+3] <- foldsIn[i] + 3
}
#Import required libraries
library(randomForest)
library(caret)
library(ROCR)
library(Boruta)
library(pROC)
library(ROCit)

#No of folds for K-fold cross validation
K=5 

#initializing variable to store classification error for the outer fold
foldsOutRFError = rep(0,K)
auc_RF= rep(0,K)

#Initializing variable to store best mtry value for each fold
mtryBest =rep(0,K)
ntreeFinal = rep(0,K)

#K fold cross validation loop
for (i in 1:K){
  
  #Splitting data into train set and validation sets
  trainSet <- x[foldsOut!=i,]
  y_trainSet <- y[foldsOut!=i]
  
  valSet <- x[foldsOut==i,]
  y_valSet <- y[foldsOut==i]
  
  #Boruta is used to identify relevant features in a dataset
  selectedFeatures <- Boruta(trainSet,y_trainSet)
  
  #vector of the names of the features that have been identified as significant
  significantFeatures <- getSelectedAttributes(selectedFeatures, withTentative = TRUE)
  
  #Using selected features for tuning & model fitting & scaling
  x_trainSet<-trainSet[,significantFeatures]
  x_valSet<-valSet[,significantFeatures]
  
  #Tuning function to get best mtry value
  tunedRFTrees<- tuneRF(x_trainSet,y_trainSet,stepFactor = 1.5, improve = 0.01, plot=FALSE,doBest = TRUE)
  
  #best mtry value corresponding to lowest OOB error estimate
  mtryBest[i] <- tunedRFTrees$mtry
  
  #fold IDs are being shuffled to remove any bias in division of data into folds
  folds_inn=sample(foldsIn)
  
  #No of inner folds
  k=4
  
  #Initializing variable to store inner fold classification error
  rf.fold.error = matrix(0,nrow=k,ncol=10)
  
  #Inner K fold cross validation for finding best ntree value 
  for(j in 1:k){
    #Dividing training set further into train set and validation sets
    x2_train <-x_trainSet[folds_inn != j,]
    y2_train <- y_trainSet[folds_inn != j]
    
    x2_valset <- x_trainSet[folds_inn == j,]
    y2_valset <- y_trainSet[folds_inn == j]
    
    #Iteration over no of trees from 100 to 1000
    for(tree in seq(100, 1000,by=100)){
      
      #random forest model fitting    
      RFfit = randomForest(x=x2_train,y=y2_train,mtry=mtryBest[i],ntree = tree, importance=TRUE)
      
      #making predictions for validation set
      prediction =predict(RFfit,x2_valset)
      
      #computing and storing fold error
      rf.fold.error[j,tree/100] <- mean(prediction!=y2_valset)
    }
  }
  #ntree value corresponding to lowest cv error
  index_ntreeFinal <-which(rf.fold.error ==min(rf.fold.error),arr.ind = TRUE)
  ntreeFinal[i] <-index_ntreeFinal[1,2]*100
  
  #model refitting with best ntree and bets mtry values
  new_RFfit = randomForest(x=x_trainSet,y=y_trainSet,mtry=mtryBest[i],ntree =ntreeFinal[i], importance=TRUE)
  
  #Make predictions with optimized model and Collecting the average CV error and AUC score
  new_pred = predict(new_RFfit,x_valSet)
  foldsOutRFError[i] <-mean(new_pred != y_valSet)
  auc_RF[i] <-auc(y_valSet, as.numeric(new_pred))
  
  #Plot ROC curve
  ROCit_obj <- rocit(score=as.numeric(new_pred),class=y_valSet)
  plot(ROCit_obj)
}
#Mean of cross validation error across K folds
cvError_mean_RF <- mean(foldsOutRFError)
cvError_mean_RF

varImpPlot(new_RFfit)

#Mean AUC score of random forest model
RF_meanAUC <- mean(auc_RF)
RF_meanAUC

#
#------------------------- 6. Quadratic Discriminant Analysis ------------------------#
#
K=5
QDAFoldError=rep(0,K)
auc_QDA = rep(0,K)
for(j in 1:K){
  
  #Splitting into training and validation set
  trainSet <- x_logReg[foldsOut!=j,]
  y_trainSet <- y[foldsOut!=j]
  
  x_valSet <- x_logReg[foldsOut==j,]
  y_valSet <- y[foldsOut==j]
  
  #Take top 50 variables highly correlated with y
  cor(trainSet,as.numeric(y_trainSet))
  indxs=order(abs(cor(trainSet,as.numeric(y_trainSet))),decreasing =TRUE)[1:50]
  x_trainSet <- trainSet[,indxs]
  
  train=data.frame(x_trainSet,y_trainSet)
  val=data.frame(x_valSet,y_valSet)
  
  #Fit QDA model
  QDAFit=qda(y_trainSet~.,data=train)
  
  #Predict on val set and calculate cross-val fold error
  QDAPredVal=predict(QDAFit,newdata=val)
  QDAFoldError[j]=mean(QDAPredVal$class!=y_valSet)
  auc_QDA[j] <- auc(y_valSet,as.numeric(QDAPredVal$class))
  
}

cvError_mean_QDA = mean(QDAFoldError)
cvError_mean_QDA

QDA_meanAUC <-mean(auc_QDA)
QDA_meanAUC


