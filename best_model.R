
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

#
#----------KNN was found with lowest error-------------
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








save(ynew, test_error,file="24.RData")