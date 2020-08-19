import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# reading a csv file using pandas library
ed=pd.read_csv("E:\Data\Assignments\i made\SLR\emp_data.csv")
ed.columns

plt.hist(ed.Salary_hike)
plt.boxplot(ed.Salary_hike,0,"rs",0)


plt.hist(ed.Churn_out_rate)
plt.boxplot(ed.Churn_out_rate)

plt.plot(ed.Salary_hike,ed.Churn_out_rate,"bo");plt.xlabel("Salary_hike");plt.ylabel("Churn_out_rate")


ed.Churn_out_rate.corr(ed.Salary_hike) # # correlation value between X and Y


np.corrcoef(ed.Churn_out_rate,ed.Salary_hike) #USING NUMPY MODULE IN MATRIX FORMAT

# linear regression model 
import statsmodels.formula.api as smf
modeled = smf.ols("Churn_out_rate~Salary_hike",data=ed).fit()
modeled
# For getting coefficients of the varibles used in equation
modeled.params

# P-values for the variables and R-squared value for prepared model
modeled.summary()

modeled.conf_int(0.05) # 95% confidence interval

prediction = modeled.predict(ed.iloc[:,0]) 
prediction
