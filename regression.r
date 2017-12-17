library(tidyverse)
library(broom)
library(glmnet)
setwd('C:/Users/mwija/OneDrive/Documents/GitHub/Data_Science_Project/')
myData <- read.csv('./data/wine_oversampled.csv')
y <- myData$quality

x <- myData[,-c(12:15)]
x = as.matrix(sapply(x, as.numeric))

lambdas <- 10^seq(3, -2, by = -.1)
fit <- glmnet(x,y,alpha=0, lambda=lambdas)
summary(fit)

cv_fit <- cv.glmnet(x, y, alpha=0, lambda=lambdas)
plot(cv_fit)

opt_lambda = cv_fit$lambda.min
fit <- cv_fit$glmnet.fit
summary(fit)

y_predicted <- predict(fit, s=opt_lambda, newx=x)
sst <- sum((y-mean(y))^2)
sse <- sum((y_predicted - y)^2)
rsq <- 1 - sse/sst
rsq
