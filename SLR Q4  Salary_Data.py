import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

slrdata=pd.read_csv("D:/ExcelR Data/Assignments/Simple linear Regration/Salary_Data.csv")

slrdata.columns

plt.hist(slrdata.YearsExperience)
plt.boxplot(slrdata.YearsExperience)
plt.plot(slrdata.YearsExperience,slrdata.Salary,"ro");plt.xlabel("YearsExperience");plt.ylabel("Salary")
plt.hist(slrdata.Salary)
plt.boxplot(slrdata.Salary)


slrdata.corr() #it should be >0.85
slrdata.Salary.corr(slrdata.YearsExperience)
np.corrcoef(slrdata.Salary,slrdata.YearsExperience)

import statsmodels.formula.api as smf
model=smf.ols("Salary~YearsExperience",data=slrdata).fit()
type(model)
# For getting coefficients of the varibles used in equation
model.params
# P-values for the variables and R-squared value for prepared model
model.summary()

model.conf_int(0.05) # 95% confidence interval

pred = model.predict(slrdata)
pred.corr(slrdata.Salary)

import matplotlib.pyplot as plt
plt.scatter(x=slrdata['YearsExperience'],y=slrdata['Salary'],color='red');plt.plot(slrdata['Salary'],pred,color='black');plt.xlabel('YearsExperience');plt.ylabel('Salary')



model2 = smf.ols('Salary~np.log(YearsExperience)',data=slrdata).fit()
model2.params
model2.summary()
print(model2.conf_int(0.01)) # 99% confidence level
pred2 = model2.predict(slrdata)
pred2.corr(slrdata.Salary)
# pred2 = model2.predict(wcat.iloc[:,0])
pred2
plt.scatter(x=slrdata['YearsExperience'],y=slrdata['Salary'],color='red');plt.plot(slrdata['Salary'],pred2,color='black');plt.xlabel('YearsExperience');plt.ylabel('Salary')

model3 = smf.ols('YearsExperience~np.log(Salary)',data=slrdata).fit()
model3.params
model3.summary()
print(model2.conf_int(0.01)) # 99% confidence level
pred3 = model3.predict(slrdata)
pred3.corr(slrdata.Salary)
# pred2 = model2.predict(wcat.iloc[:,0])
pred3
plt.scatter(x=slrdata['YearsExperience'],y=slrdata['Salary'],color='red');plt.plot(slrdata['Salary'],pred2,color='black');plt.xlabel('YearsExperience');plt.ylabel('Salary')
