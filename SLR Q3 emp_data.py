import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

empdata=pd.read_csv("D:/ExcelR Data/Assignments/Simple linear Regration/emp_data.csv")

empdata.columns

plt.hist(empdata.Salary_hike)
plt.boxplot(empdata.Salary_hike)
plt.plot(empdata.Salary_hike,empdata.Churn_out_rate,"ro");plt.xlabel("Salary_hike");plt.ylabel("Churn_out_rate")
plt.hist(empdata.Churn_out_rate)
plt.boxplot(empdata.Churn_out_rate)


empdata.corr()
empdata.Churn_out_rate.corr(empdata.Salary_hike)
np.corrcoef(empdata.Churn_out_rate,empdata.Salary_hike)

import statsmodels.formula.api as smf
model=smf.ols("Churn_out_rate~Salary_hike",data=empdata).fit()
type(model)
# For getting coefficients of the varibles used in equation
model.params
# P-values for the variables and R-squared value for prepared model
model.summary()

model.conf_int(0.05) # 95% confidence interval

pred = model.predict(empdata)
