rm(list=ls())
# Data Import ------------------------------------------------------------
library(caret)
library(klaR)
library(lattice)
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

#PART A
#we now consider a linear model for our data (LOGED)
bruhnilda.lm <- lm(Sulfate ~ Hours, data = bruhnilda)
plot(bruhnilda, xlab = "LOG_HOURS", ylab = "LOG_SULFATE", main = "LOGED x and y values Linear Regression Model")
abline(bruhnilda.lm, col="red")
summary(bruhnilda.lm) 

#PART B 
#we now consider a linear model for our data (ORIGINAL)
bruhnilda_orig.lm <- lm(Sulfate ~ Hours, data = bruhnilda_orig)
plot(bruhnilda_orig, xlab = "HOURS", ylab = "SULFATE", main = "Original x and y values Linear Regression Model")
abline(bruhnilda_orig.lm, col="red")
summary(bruhnilda_orig.lm) 
#clearly lower Rsquared value and a really bad abline approximation, the variable 
#transformation has clearly helped

#PART C 

# "Rather than stopping here we perform some investigations using residual 
# diagnostics to determine whether the various assumptions that underpin 
# linear regression are reasonable for our data or if there is evidence 
# to suggest that additional variables are required in the model or some 
# other alterations to identify a better description of the variables 
# that determine how weight changes."

#we will use the lattice library 

# residual against fitted values in log-log coordinates
xyplot(resid(bruhnilda.lm) ~ fitted(bruhnilda.lm),
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


# residual against fitted values in original coordinates
xyplot(resid(bruhnilda_orig.lm) ~ fitted(bruhnilda_orig.lm),
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


