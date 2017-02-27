rm(list=ls())
library(caret)
library(data.table)
library(klaR)
library(lattice)
library(faraway)
library(readr)
# setwd('/Users/brunocalogero/Desktop/UIUC/cs498aml/cs498aml/hw4/7.11')

#pre-processing includes making csv file for easier manipulation 
# it also includes adding extra 10th collumn named "age" adding 1.5 to ring values to convert to age
raw_data <- read_csv('abalone.csv')
data_no_gender <- raw_data[,-c(1)]

#PART A 
#linear regression predicting the age from the measurements, ignoring gender
model1 <- lm(age ~ 1 + length + diameter + height + whole_weight + shuckled_weight + viscera_weight + shell_weight + rings, data = data_no_gender)

xyplot(resid(model1) ~ fitted(model1),
       xlab = "Fitted Values",
       ylab = "Residuals",
       main = "Residual Diagnostic Plot",
       panel = function(x, y, ...)
       {
         panel.grid(h = -1, v = -1)
         panel.abline(h = 0)
         panel.xyplot(x, y, ...)
       }
)


#PART B 
#linear regression predicting the age from the measurements, including gender

#data manipulation converting male to 1, female to 0 and rest to -1
data_with_gender <- raw_data 
data_with_gender[data_with_gender=="M"]<- 1 
data_with_gender[data_with_gender=="F"]<- 0
data_with_gender[data_with_gender=="I"]<- -1

model2 <- lm(age ~ 1 + sex + length + diameter + height + whole_weight + shuckled_weight + viscera_weight + shell_weight + rings, data = data_with_gender)

xyplot(resid(model2) ~ fitted(model2),
       xlab = "Fitted Values",
       ylab = "Residuals",
       main = "Residual Diagnostic Plot",
       panel = function(x, y, ...)
       {
         panel.grid(h = -1, v = -1)
         panel.abline(h = 0)
         panel.xyplot(x, y, ...)
       }
)