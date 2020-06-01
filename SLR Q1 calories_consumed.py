import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

wgcc=pd.read_csv("D:/ExcelR Data/Assignments/Simple linear Regration/calories_consumed.csv")

wgcc.columns

plt.hist(wgcc.Weight_gained)
plt.boxplot(wgcc.Weight_gained)
plt.plot(wgcc.Weight_gained,wgcc.Calories_Consumed,"ro");plt.xlabel("Weight_gained");plt.ylabel("Calories_Consumed")
plt.hist(wgcc.Calories_Consumed)
plt.boxplot(wgcc.Calories_Consumed)


wgcc.corr()
wgcc.Calories_Consumed.corr(wgcc.Weight_gained)
np.corrcoef(wgcc.Calories_Consumed,wgcc.Weight_gained)

import statsmodels.formula.api as smf
model=smf.ols("Weight_gained~Calories_Consumed",data=wgcc).fit()
type(model)
# For getting coefficients of the varibles used in equation
model.params
# P-values for the variables and R-squared value for prepared model
model.summary()

model.conf_int(0.05) # 95% confidence interval

pred = model.predict(wgcc)
