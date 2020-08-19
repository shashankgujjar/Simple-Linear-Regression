import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# reading a csv file using pandas library
cc=pd.read_csv("E:\Data\Assignments\i made\SLR\calories_consumed.csv")
cc.columns

plt.hist(cc.Weight_gained_grams)
plt.boxplot(cc.Weight_gained_grams,0,"rs",0)


plt.hist(cc.Calories_Consumed)
plt.boxplot(cc.Calories_Consumed)

plt.plot(cc.Weight_gained_grams,cc.Calories_Consumed,"bo");plt.xlabel("Weight_gained_grams");plt.ylabel("Calories_Consumed")


cc.Calories_Consumed.corr(cc.Weight_gained_grams) 
# # correlation value between X and Y
# 0.9469910088554458
# the value shows high correlation between the variables

np.corrcoef(cc.Calories_Consumed,cc.Weight_gained_grams) 
#USING NUMPY MODULE IN MATRIX FORMAT



# linear regression model 
import statsmodels.formula.api as smf
model = smf.ols("Calories_Consumed~Weight_gained_grams",data=cc).fit()
model

# For getting coefficients of the varibles used in equation
model.params

# P-values for the variables and R-squared value for prepared model
model.summary()

model.conf_int(0.05) # 95% confidence interval

pred = model.predict(cc.iloc[:,0]) # Predicted values of Calories_Consumed using the model
pred


error = cc['Calories_Consumed'] - pred
