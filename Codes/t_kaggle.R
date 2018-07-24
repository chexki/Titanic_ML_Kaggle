### Predict Survival Of Passengers On The Titanic Ship_ (www.kaggle.com)
# By Chetan Jawale.###

### Loading Dataset

ttrain <- read.csv(file.choose())
ttest <- read.csv(file.choose())
#ttest$Survived <- c(0,1)

## Feature selection

library(ggplot2)
ggplot(ttrain, aes(x=reorder(SibSp ,SibSp, function(x)-length(x)), fill = Survived)) + 
  geom_bar() + geom_bar() +
  xlab('SibSp') + ylab('count') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(ttrain, aes(x=reorder(Parch ,Parch, function(x)-length(x)), fill = Survived)) + 
  geom_bar() + geom_bar() +
  xlab('Parch') + ylab('count') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

ggplot(ttrain, aes(x=reorder(Embarked ,Embarked, function(x)-length(x)), fill = Survived)) + 
  geom_bar() + geom_bar() +
  xlab('Embarked') + ylab('count') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# PassengerID , Name, Cabin and Ticket are some features which does not add any information
# to reach our objective of predicting survival of passenger

# Hence, such features does not be needed.

str(train)
train <- ttrain[,c("Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Survived")]
test <- ttest[,c("Pclass","Sex","Age","SibSp","Parch","Fare","Embarked")]

##################################################################################################
### Checking if missing values are present

sapply(train,function(x) sum(is.na(x)))
sapply(test,function(x) sum(is.na(x)))

# Fill in missing values for age by mean and Fare by median

train$Age[is.na(ttrain$Age)] = mean(ttrain$Age, na.rm=TRUE)
test$Age[is.na(ttest$Age)] = mean(ttest$Age, na.rm =TRUE)

summary(train$Fare)
summary(test$Fare)

test$Fare[is.na(test$Fare)] = median(test$Fare, na.rm =TRUE)
################################################################################################

## Pcclass from integer to categorical variable

train$Pclass = as.factor(train$Pclass)
test$Pclass = as.factor(test$Pclass)

train$SibSp = as.factor(train$SibSp)
test$SibSp = as.factor(test$SibSp)

train$Parch = as.factor(train$Parch)
test$Parch = as.factor(test$Parch)

str(train)
summary(train)

train$Embarked[train$Embarked==''] <- 'S'
train$Embarked = as.character(train$Embarked)
train$Embarked = as.factor(train$Embarked)
summary(train)


str(test)
summary(test)
levels(test$Parch)
levels(train$Parch)
test$Parch[test$Parch== 9] <- 0
test$Parch = as.character(test$Parch)
test$Parch = as.factor(test$Parch)
summary(test$Parch)


## Model building
# Generelised Linear Model (Logistic Regression technique)

model = glm(Survived ~., family = binomial, data = train)
summary(model)
library(car)
vif(model)

## Variable significance selection

model1 = step(glm(Survived ~., family = binomial, data = train),direction = "both")
summary(model1)
vif(model1)

## Giving Own Reference

str(train)
table(train$Pclass)
table(train$Sex)
table(train$SibSp)
table(train$Embarked)
table(train$Parch)

model2 = step(glm(Survived ~relevel(Pclass,ref = 2)+relevel(Sex,ref ='male')+
                    relevel(SibSp,ref = 5)+relevel(Parch,ref = 6)+relevel(Embarked,ref ='Q')+
                    Age+Fare, family = binomial, data = train),direction = "both")
summary(model2)
vif(model2)

#
library(oddsratio)
library(vcd)
mytable<- table(train$Survived,train$Sex);mytable

# Females are more likely to be survived than males

oddsratio(mytable,log = F)
summary(test)

###Predicton for model 

tpredict<- predict(model2, newdata = test, type = 'response')
pred.logit <- rep('0',length(tpredict))
pred.logit[tpredict>=0.5] <- '1'
pred.logit

# Accuracy

Submission1 <-data.frame(PassengerId = ttest$PassengerId, Survived = pred.logit)
write.csv((Submission1), "Submission1.csv", row.names = FALSE)

# we get an accuracy upto 0.76555 on kaggle submission.

######################################################################################################

## Random Forest Model

library("randomForest")
set.seed(370)

fitrfmodel = randomForest(Survived~.,data = train, 
                          ntree = 700, 
                          importance = TRUE)
summary(fitrfmodel)

par(mfrow=c(1,1))
plot(fitrfmodel)

# K-FOLD cross validation of the model

library(e1071)
set.seed(1)
tune.out = tune(randomForest, Survived~., data = train, kernel ="binomial",
                ranges = list(cost  = c(0.001, 0.01, 0.1, 1, 5, 10,100)))

summary(tune.out)

# Classification Tree with rpart

library(rpart)
fol= formula( as.factor(Survived) ~ Pclass + Age + Sex + Parch + SibSp + Fare + Embarked)
fit <- rpart( fol, data=train, method= "class")
plot(fit, uniform=TRUE, main="Classification Tree")    
text(fit, use.n=FALSE, all=TRUE, cex=.8 )  

# Best Model
bestmod_rf = tune.out$best.model
summary(bestmod_rf)
plot(bestmod_rf)

# Accuracy Of the model

tpredict_rf <- predict(bestmod_rf, newdata = test, type = 'response')
pred.logit_rf <- rep('0',length(tpredict_rf))
pred.logit_rf[tpredict_rf>=0.5] <- '1'
pred.logit_rf

# Accuracy

Submission2 <-data.frame(PassengerId = ttest$PassengerId, Survived = pred.logit_rf)
write.csv((Submission2), "Submission2.csv", row.names = FALSE)

# we get an accuracy upto 0.75598 on kaggle submission.

####################################################################################################

# Support Vector Machines

str(train)
train1 <- train[c("Pclass","Sex","Age","Fare","Survived")]

svm_model <- svm(Survived ~.,data = train1, kernal ="binomial")
summary(svm_model)

# Accuracy Of the model

tpredict_svm <- predict(svm_model, newdata = test, type = 'response')
pred.logit_svm <- rep('0',length(tpredict_svm))
pred.logit_svm[tpredict_svm>=0.5] <- '1'
pred.logit_svm

# Accuracy

Submission34 <-data.frame(PassengerId = ttest$PassengerId, Survived = pred.logit_svm)
write.csv((Submission34), "Submission34.csv", row.names = FALSE)

# we get an accuracy upto 0.77033 on kaggle submission.
