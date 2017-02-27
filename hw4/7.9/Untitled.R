rm(list=ls())
# Data Import ------------------------------------------------------------
library(caret)
library(klaR)
setwd('/Users/brunocalogero/Desktop/UIUC/cs498aml/cs498aml/hw4/7.9')

raw_data <- fread('data.txt')

#preparing data 
hours <- raw_data[,c(1)] 
sulfate <- raw_data[,c(2)] 

#ln or log10 
hours_log <-log(hours)
sulfate_log <- log(sulfate)

#create data frames for our analysis 
bruhnilda_orig <- data.frame(hours,sulfate)
bruhnilda <- data.frame(hours_log,sulfate_log) 

#we now consider a linear model for our data (LOGED)
bruhnilda.lm <- lm(Hours ~ Sulfate, data = bruhnilda)
plot(bruhnilda, xlab = "LOG_HOURS", ylab = "LOG_SULFATE", main = "LOGED x and y values Linear Regression Model")
abline(bruhnilda.lm, col="red")
summary(bruhnilda.lm) 


#we now consider a linear model for our data (ORIGINAL)
bruhnilda_orig.lm <- lm(Hours ~ Sulfate, data = bruhnilda_orig)
plot(bruhnilda_orig, xlab = "HOURS", ylab = "SULFATE", main = "Original x and y values Linear Regression Model")
abline(bruhnilda_orig.lm, col="red")
summary(bruhnilda_orig.lm) 
#clearly lower Rsquared value and a really bad abline approximation, the variable 
#transformation has clearly helped



