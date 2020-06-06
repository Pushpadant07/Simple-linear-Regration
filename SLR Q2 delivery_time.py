import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

dltm=pd.read_csv("D:/ExcelR Data/Assignments/Simple linear Regration/delivery_time.csv")

dltm.columns

plt.hist(dltm.Delivery_Time)
plt.boxplot(dltm.Delivery_Time)
plt.plot(dltm.Delivery_Time,dltm.Sorting_Time,"ro");plt.xlabel("Delivery_Time");plt.ylabel("Sorting_Time")
plt.hist(dltm.Sorting_Time)
plt.boxplot(dltm.Sorting_Time)


dltm.corr()
dltm.Sorting_Time.corr(dltm.Delivery_Time)
np.corrcoef(dltm.Sorting_Time,dltm.Delivery_Time)

import statsmodels.formula.api as smf
model=smf.ols("Delivery_Time~Sorting_Time",data=dltm).fit()
type(model)
# For getting coefficients of the varibles used in equation
model.params
# P-values for the variables and R-squared value for prepared model
model.summary()
# if the R squre value is in between 0.6-0.8 then we can say that our model is good one  
model.conf_int(0.05) # 95% confidence interval
#if p value is >0.05 then only go for conf int 
pred = model.predict(dltm)
