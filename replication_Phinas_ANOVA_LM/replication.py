#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 10:16:43 2020

@author: andrea

Replication of the following results
https://link.springer.com/article/10.3758/s13428-011-0172-y/tables/2

"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


#recreating the dataframe
array1 = np.array([[1, 489.33, 'Cond1'],[1, 454.82, 'Cond2'],
       [1, 425.07, 'Cond3'],
       [1, 450.1, 'Cond4'],
       [1, 395.0, 'Cond5'],
       [2, 412.83, 'Cond1'],
       [2, 398.87, 'Cond2'],
       [2, 408.87, 'Cond3'],
       [2, 390.5, 'Cond4'],
       [2, 375.7, 'Cond5'],
       [3, 489.23, 'Cond1'],
       [3, 458.98, 'Cond2'],
       [3, 423.73, 'Cond3'],
       [3, 410.25, 'Cond4'],
       [3, 397.0, 'Cond5'],
       [4, 549.21, 'Cond1'],
       [4, 472.07, 'Cond2'],
       [4, 451.57, 'Cond3'],
       [4, 431.05, 'Cond4'],
       [4, 419.7, 'Cond5'],
       [5, 459.64, 'Cond1'],
       [5, 428.1, 'Cond2'],
       [5, 403.1, 'Cond3'],
       [5, 393.53, 'Cond4'],
       [5, 372.2, 'Cond5'],
       [6, 467.09, 'Cond1'],
       [6, 438.6, 'Cond2'],
       [6, 393.33, 'Cond3'],
       [6, 367.6, 'Cond4'],
       [6, 417.9, 'Cond5'],
       [7, 424.22, 'Cond1'],
       [7, 381.28, 'Cond2'],
       [7, 388.97, 'Cond3'],
       [7, 385.84, 'Cond4'],
       [7, 362.8, 'Cond5'],
       [8, 377.05, 'Cond1'],
       [8, 351.67, 'Cond2'],
       [8, 346.2, 'Cond3'],
       [8, 339.7, 'Cond4'],
       [8, 343.6, 'Cond5']])

#convert the array into a dataframe
dataset = pd.DataFrame({'Subj': array1[:, 0], 'RT': array1[:, 1], 'Cond': array1[:, 2]})
dataset["RT"] = pd.to_numeric(dataset["RT"]) #convert values to numbers

subjects = 8 #subjects from the paper

#ONE-WAY REPEATED MEASURES ANOVA
#Print results with relevant SSs MS df F p
lmfit = ols('RT ~ C(Cond)+C(Subj)',data=dataset).fit()
results_summary_1 = sm.stats.anova_lm(lmfit, typ = 2)
df_means = dataset.groupby(["Cond"])['RT'].agg(['mean', 'std']) #calculate means

ms_participant = (results_summary_1.iat[1,0])/(results_summary_1.iat[1,1])                                                    
ms_cond = (results_summary_1.iat[0,0])/(results_summary_1.iat[0,1])                                                    
ms_participant_x_cond = (results_summary_1.iat[2,0])/(results_summary_1.iat[2,1]) 
results_summary_1.insert(2, "MS", [ms_cond, ms_participant, ms_participant_x_cond], True) 
results_summary_1.loc["C(Subj)",'F'] = ""
results_summary_1.loc["Residual",'F'] = ""

ss_cond = results_summary_1.iat[0,0]
ss_error = results_summary_1.iat[2,0]
results_summary_1['η2p'] = (ss_cond)/(ss_cond + ss_error), "", ""

#now the fitted data 
#convert Cond to numeric 
dataset['Cond_L'] = pd.to_numeric(dataset['Cond'].str.replace("Cond","")) 
#Another 1-way ANOVA
lmfit = ols('RT ~ Cond_L*C(Subj)',data=dataset).fit()                                                   
results_summary_2 = sm.stats.anova_lm(lmfit, typ = 2)
results_summary_2 = results_summary_2.drop(['C(Subj)', 'Residual'])

#get SS Linear 
ss_linear = results_summary_2.iat[0,0]
ss_linear_x_participant = results_summary_2.iat[1,0]

p_e_sq_2 = ss_linear/(ss_linear_x_participant + ss_linear)
           
ms_linear_x_participant = ss_linear_x_participant/(subjects-1)                                                    
F = (ss_linear/ ms_linear_x_participant)

results_summary_2.insert(2, "MS", [ss_linear, ms_linear_x_participant], True) 
results_summary_2.at["Cond_L","F"] = F
results_summary_2.loc["Cond_L:C(Subj)",'F'] = ""

results_summary_2['η2p'] = p_e_sq_2, ""

frames = [results_summary_1, results_summary_2]
result = pd.concat(frames)
result = result.drop(['PR(>F)'], axis=1)


slope = ss_linear/(8*((2*2)+(1*1)+0+(1*1)+(2*2)))
slope = slope**(1/2)

print(result, "\n*p<0.05\n\nslope is:", slope)

import seaborn as sns
import matplotlib.pyplot as plt
ax = sns.regplot(x='Cond_L', y='RT', data=dataset, x_estimator=np.mean, color="y")
ax.set(ylabel='Reaction Times (RT)', xlabel='Distance')
ax.set(xticks=np.arange(1, 6, 1)) #limit the number of ticks to 5
#ax.set_xticks(range(len(dataset))) # <--- set the ticks first
ax.set_xticklabels(['Cond1','Cond2','Cond3', "Cond4", "Cond5"])
plt.savefig('saving-a-seaborn-plot-as-pdf-file-300dpi.pdf', dpi = 300)




