setwd('C:/Users/mwija/OneDrive/Documents/GitHub/Data_Science_Project/')
myData <- read.csv('./data/wine_oversampled_R.csv')


wine <- myData[,c(0:13)]
wine = as.matrix(sapply(wine, as.numeric))
wineDataFrame <- as.data.frame(wine)

#assign factor level to white wine and red wine
wineDataFrame$color<- factor(wineDataFrame$color, levels=c(0,1), labels = "white","red")

#Multiple linear regression of beta1x1 + beta2x2+...
model1<- lm(quality ~ .,data = wineDataFrame)
summary(model1)

#Multiple linear regression with all of the combination of interaction term
model2 <- lm(quality ~.^2, data=wineDataFrame) #R^2 = 0.3794

#Update model2 by taking off the most non significant interaction term until all has exhausted
model3 <- update(model2, .~. - free.sulfur.dioxide:sulphates) #R^2 = 0.3794
model4 <- update(model3, .~. - fixed.acidity:total.sulfur.dioxide) #R^2 = 0.3794
model5 <- update(model4, .~. - fixed.acidity:density)
model6 <- update(model5, .~. - density:pH)
model7 <- update(model6, .~. - free.sulfur.dioxide:pH)
model8 <- update(model7, .~. - chlorides:alcohol)
model9 <- update(model8, .~. - volatile.acidity:sulphates) #R^2 = 0.3793
model10 <- update(model9, .~. - chlorides:free.sulfur.dioxide)
model11 <- update(model10, .~. - citric.acid:total.sulfur.dioxide)
model12 <- update(model11, .~. - total.sulfur.dioxide:pH)
model13 <- update(model12, .~. - volatile.acidity:pH)
model14 <- update(model13, .~. - volatile.acidity:free.sulfur.dioxide) #R^2 = 0.3792
model15 <- update(model14, .~. - residual.sugar:chlorides)
model16 <- update(model15, .~. -residual.sugar:sulphates) #R^2 = 0.3791
model17 <- update(model16, .~. - sulphates:alcohol)
model18 <- update(model17, .~. -density:sulphates)
model19 <- update(model18, .~. - fixed.acidity:sulphates)
model20 <- update(model19, .~. - pH:alcohol) #R^2 = 0.379
model21 <- update(model20, .~. -citric.acid:sulphates)
model22 <- update(model21, .~. -residual.sugar:alcohol)
model23 <- update(model22, .~. -volatile.acidity:chlorides) #R^2 = 0.3787
model24 <- update(model23, .~. - chlorides:total.sulfur.dioxide)
model25 <- update(model24, .~. - fixed.acidity:free.sulfur.dioxide)
#the best model we can have since the interaction term are significant, we can't remove the individual term like pH, sulphates
summary(model25)
