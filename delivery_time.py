import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# reading a csv file using pandas library
dt=pd.read_csv("E:\Data\Assignments\i made\SLR\delivery_time.csv")
dt.columns

plt.hist(dt.Delivery_Time)
plt.boxplot(dt.Delivery_Time,0,"rs",0)


plt.hist(dt.Sorting_Time)
plt.boxplot(dt.Sorting_Time)

plt.plot(dt.Delivery_Time,dt.Sorting_Time,"bo");plt.xlabel("Sorting_Time");plt.ylabel("Dlivery_Time")


dt.Delivery_Time.corr(dt.Sorting_Time) # # correlation value between X and Y


np.corrcoef(dt.Sorting_Time,dt.Delivery_Time) #USING NUMPY MODULE IN MATRIX FORMAT

# linear regression model
import statsmodels.formula.api as smf
model = smf.ols("Sorting_Time~Delivery_Time",data=dt).fit()
model
# For getting coefficients of the varibles used in equation
model.params

# P-values for the variables and R-squared value for prepared model
model.summary()

model.conf_int(0.05) # 95% confidence interval

pred = model.predict(dt.iloc[:,0]) # Predicted values of Calories_Consumed using the model
pred
