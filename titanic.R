#titanic dataset data mining

setwd("C:\\Users\\Samuel\\Documents\\R\\Excel Practice Sheets\\kaggle") # Set my working directory

# Load packages
library(ggplot2)
library(ggthemes)
library(randomForest) 
library(dplyr)

set.seed(1)
train <- read.csv("train.csv", header = TRUE)
test <- read.csv("test.csv", header = TRUE)
full <- bind_rows(train, test)
names(test)
dim(test1)


#ticket is nonsense, as is cabin because so much information is missing. name is also dropped. there could very well be a relationship between 

#Age and Survived
library(fields)
bplot.xy(train$Survived, train$Age)
summary(train$Survived) #quite a few NAs

#look at fares
bplot.xy(train$Survived, train$Fare)


extractFeatures <- function(data) {
  features <- c("Pclass",
                "Age",
                "Sex",
                "Parch",
                "SibSp",
                "Fare",
                "Embarked")
  fea <- data[,features]
  fea$Age[is.na(fea$Age)] <- -1
  fea$Fare[is.na(fea$Fare)] <- median(fea$Fare, na.rm=TRUE)
  fea$Embarked[fea$Embarked==""] = "S"
  fea$Sex      <- as.factor(fea$Sex)
  fea$Embarked <- as.factor(fea$Embarked)
  return(fea)
}

train1 <-extractFeatures(train)
test1 <- extractFeatures(test)
train1$Survived <- as.factor(train$Survived)
test1$Survived <- test$Survived

levels(test1$Survived) <- levels(train1$Survived)
names(train1)
names(test1)

test1 <- rbind(train1[1, ] , test1) # shortcut to verify that all columns match
test1 <- test1[-1,]

rf <- randomForest(as.factor(train1$Survived) ~., data=train1,  mtry=7, importance =TRUE, ntree = 100)
pred.rf = predict(rf ,newdata =test1)
summary(pred.rf)
#MSE = 11 which is improvement





#a loop to consider other values of mtry
class =matrix(NA, 7,2)
for(i in 1:7){
  fit=randomForest(Survived~.,data=train1, mtry=i, ntree=100)
  pred=predict(fit,newdata =test1)
  class[i,] = summary(pred)
}
class


varImpPlot(rf)


importance <- importance(rf)
varImportance <- data.frame(Variables = row.names(importance), Importance = importance[,1])


#plot the importance
ggplot(varImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  labs(x = 'Variables') +
  coord_flip() + 
  theme_solarized()


#embarked is clearly the least important, so I'll omit it
rf <- randomForest(as.factor(train1$Survived) ~., data=train1,  mtry=1, importance =TRUE, ntree = 50)
pred.rf = predict(rf ,newdata =test1)
summary(pred.rf)
               
solution <- data.frame(PassengerID = test$PassengerId, Survived = pred.rf)               
write.csv(solution, file = "random_forest_2_submission.csv", row.names=FALSE)               