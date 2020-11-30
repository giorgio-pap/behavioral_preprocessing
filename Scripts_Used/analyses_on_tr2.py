#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:37:41 2020

@author: papitto
"""

import pandas as pd
import os
import re
import numpy as np 
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols

#avoid warning
import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    
data_folder_training ="/data/pt_02312/MRep_training_backup/data/"

#create two empty dataframe where to store the results
df_result_tr = pd.DataFrame([]) #empty dataframe for result file

#read all the files in a folders and creat a list
filenames_tr = os.listdir(data_folder_training)

#create a list only with files ending with the word "UPDATED"
#first create an empy list for Training and Test
UPDATED_files_tr = []

#append the filenames to the lists
for filename_tr in filenames_tr:
    if filename_tr.endswith("UPDATED.csv"):
        UPDATED_files_tr.append(filename_tr)


#sort the filenames in some order
UPDATED_files_tr = sorted(UPDATED_files_tr)

#extract the number of the participants for Tr
pattern = re.compile(r'(\d*)_Mental')
participant_numbers =  ["; ".join(pattern.findall(item)) for item in UPDATED_files_tr]

df_tr_2_tot = []

for UPDATED_file_tr in UPDATED_files_tr: 

    participant_number_tr = [int(s) for s in re.findall(r'(\d*)_Mental', UPDATED_file_tr)]
    str_number = ' '.join(map(str, participant_number_tr)) 

    df_tr = pd.read_csv(data_folder_training + UPDATED_file_tr,  header=0) #read the data file in
    
    #only keep rows referring to trials (experimental or filler)
    df_exp_fil_trials = df_tr.loc[(df_tr['trial_type'] == "experimental")|(df_tr['trial_type'] == "filler")]
    
    group = df_exp_fil_trials["group"].iloc[0] #extract group information (A, B, C, or D)
   
    ###############################
    ######### TRAINING 2 #########
    ##############################
    
    #repeat exactly the same thing but for training 2
    # "repeat_training_loop1b.thisRepN" is the value assigned in the original file to training 2 loopes
    df_training_2 = df_exp_fil_trials.loc[df_tr['repeat_training_loop1b.thisRepN'] >= 0] 
    df_training_2 = df_training_2.loc[df_tr['resp_total_corr'] == 1] 
    Total_resp_Seq_2 = df_training_2["resp_total_corr"].sum()
    
    df_training_2_wo = df_exp_fil_trials.loc[df_tr['repeat_training_loop1b.thisRepN'] >= 0]
    df_training_2_wo = df_training_2_wo.loc[df_tr['resp_total_corr'] == 1] 
    df_training_2_wo = df_training_2_wo.drop(df_training_2_wo[df_training_2_wo['trial_type'] == "filler"].index)
    Corr_S2_Tot_wo = df_training_2_wo["resp_total_corr"].sum() #number of correct sequences (without fillers) - all loops


    df_tr_2_tot.append(df_training_2_wo)

  
df_tr_2_tot = pd.concat(df_tr_2_tot)
df_tr_2_tot = df_tr_2_tot[["participant", "resp1b.rt", "resp_total_time", "conditions", 
                           "repeat_training_loop1b.thisRepN"]] 

df_tr_2_tot = df_tr_2_tot.rename(columns = {'participant': 'Id', 'resp1b.rt': 'OT', 
                                              'resp_total_time': 'RT', 'conditions': 'condition',
                                              "repeat_training_loop1b.thisRepN": "loop"}, inplace = False)

 
df_tr_2_tot.to_csv("/data/pt_02312/results/results_training_2.csv", index = False, header=True)




#########
result_total = df_tr_2_tot

result_total['condition'].replace(["spec"], '1_specific', inplace=True)
result_total['condition'].replace(["subrule"], '2_subrule', inplace=True)
result_total['condition'].replace(["rule"], '3_rule', inplace=True)
result_total['condition'].replace(["general"], '4_general', inplace=True)

result_total = result_total.sort_values('condition')

result_total['condition'].replace(["1_specific"], 'spec', inplace=True)
result_total['condition'].replace(["2_subrule"], 'subrule', inplace=True)
result_total['condition'].replace(["3_rule"], 'rule', inplace=True)
result_total['condition'].replace(["4_general"], 'general', inplace=True)

rddf_OT = result_total[['Id','condition', "OT"]]
rddf_RT = result_total[['Id','condition', "RT"]]


#### identifying outliers ####
## boxplots
for rt_ot in ["RT","OT"]:
    
    if rt_ot == "RT":
        df_rt_ot = rddf_RT
    elif rt_ot == "OT":
        df_rt_ot = rddf_OT
        
    pl_cond = sns.boxplot(x = "condition", y = rt_ot, data = df_rt_ot) #condition
    plt.figure()
    pl_subj = sns.boxplot(x = "Id", y = rt_ot, data = df_rt_ot) #subject

    #this loop creates a plot for each condition x id
    for cond in df_rt_ot["condition"].unique():  
        rddf_sub = df_rt_ot[df_rt_ot.condition == cond]
        plt.figure() #this creates a new figure on which your plot will appear
        pl_subj_cond = sns.boxplot(x = "Id", y = rt_ot, data = rddf_sub) # subject x condition
        pl_subj_cond.set_title(cond)
        plt.figure()
        
## IQR 
#create empty dataframes for appending and concatenating 
df_without_out_cond_conc_RT = [] 
df_without_out_subj_conc_RT = []
df_without_out_subj_cond_conc_RT = []

df_only_out_cond_conc_RT = [] 
df_only_out_subj_conc_RT = []
df_only_out_subj_cond_conc_RT = []


df_without_out_cond_conc_OT= [] 
df_without_out_subj_conc_OT = []
df_without_out_subj_cond_conc_OT = []

df_only_out_cond_conc_OT = [] 
df_only_out_subj_conc_OT = []
df_only_out_subj_cond_conc_OT = []

#define the function IQR
def iqr_func(dataframe_outliers):
    q1, q3 = np.percentile(dataframe_outliers[rt_ot], [25, 75])
    iqr = q3 - q1
    df_RT = dataframe_outliers[rt_ot] 
    #get boolean values of true_false
    outliers_T_F = (df_RT < (q1 - 1.5 * iqr)) |(df_RT > (q3 + 1.5 * iqr)) 
    df_without_out = dataframe_outliers[~(outliers_T_F)] #keep where it is false
    df_only_out = dataframe_outliers[outliers_T_F] #keep where it is true
    
    #print IQR values for all possibilities
#    if outliers_focus == "condition":
#        print("IQR for", cond, "is", iqr)
#    elif outliers_focus == "subject": 
#        print("IQR for subject", subj, "is", iqr)
#    else:
#        print("IQR for", cond, "and subject", subj, "is", iqr) 
    
    #return dataframes (1) one without outliers, (2) one only with outliers
    return df_without_out, df_only_out  

for rt_ot in ["RT","OT"]:
   
    if rt_ot == "RT":
        df_rt_ot = rddf_RT
        df_1 = df_without_out_cond_conc_RT
        df_2 = df_only_out_cond_conc_RT
        df_3 = df_without_out_subj_conc_RT
        df_4 = df_only_out_subj_conc_RT
        df_5 = df_without_out_subj_cond_conc_RT
        df_6 = df_only_out_subj_cond_conc_RT
    elif rt_ot == "OT":
        df_rt_ot = rddf_OT
        df_1 = df_without_out_cond_conc_OT
        df_2 = df_only_out_cond_conc_OT
        df_3 = df_without_out_subj_conc_OT
        df_4 = df_only_out_subj_conc_OT
        df_5 = df_without_out_subj_cond_conc_OT
        df_6 = df_only_out_subj_cond_conc_OT
        
    #CONDITIONS
    for cond in df_rt_ot["condition"].unique():
        outliers_focus = "condition"
        rddf_sub_out = df_rt_ot[df_rt_ot.condition == cond]
        df_without_out_cond, df_only_out_cond = iqr_func(rddf_sub_out) #extract the variables form the function

        df_1.append(df_without_out_cond) #append condition dataframes after each loop
        df_2.append(df_only_out_cond) 

    #SUBJECTS
    for subj in result_total["Id"].unique():
        outliers_focus = "subject"
        rddf_sub_out = result_total[result_total.Id == subj]
        df_without_out_subj, df_only_out_subj = iqr_func(rddf_sub_out)

        df_3.append(df_without_out_subj)
        df_4.append(df_only_out_subj)

    #SUBJECTS X CONDITION
    for subj in result_total["Id"].unique(): 
        for cond in result_total["condition"].unique():
            outliers_focus = "subject and condition"
            rddf_sub_out = result_total[result_total.condition == cond]
            rddf_sub_out = rddf_sub_out[rddf_sub_out.Id == subj] 
            df_without_out_subj_cond, df_only_out_subj_cond = iqr_func(rddf_sub_out)

            df_5.append(df_without_out_subj_cond) 
            df_6.append(df_only_out_subj_cond) 

def conc(output_df):
    output_df = pd.concat(output_df)
    output_df = output_df.reset_index(drop=True)   

    return output_df


df_without_out_cond_conc_RT, df_only_out_cond_conc_RT = (conc(df) for df in [df_without_out_cond_conc_RT, df_only_out_cond_conc_RT]) # or either .map or .applymap
df_without_out_cond_conc_OT, df_only_out_cond_conc_OT = (conc(df) for df in [df_without_out_cond_conc_OT, df_only_out_cond_conc_OT])

df_without_out_subj_conc_RT, df_only_out_subj_conc_RT = (conc(df) for df in [df_without_out_subj_conc_RT, df_only_out_subj_conc_RT]) # or either .map or .applymap
df_without_out_subj_conc_OT, df_only_out_subj_conc_OT = (conc(df) for df in [df_without_out_subj_conc_OT, df_only_out_subj_conc_OT])

df_without_out_subj_cond_conc_RT, df_only_out_subj_cond_conc_RT = (conc(df) for df in [df_without_out_subj_cond_conc_RT, df_only_out_subj_cond_conc_RT]) # or either .map or .applymap
df_without_out_subj_cond_conc_OT, df_only_out_subj_cond_conc_OT = (conc(df) for df in [df_without_out_subj_cond_conc_OT, df_only_out_subj_cond_conc_OT])


# boxplots to check results
for rt_ot in ["RT","OT"]:
    
    if rt_ot == "RT":
        df_rt_ot_1 = df_without_out_cond_conc_RT
        df_rt_ot_2 = df_without_out_subj_conc_RT
        df_rt_ot_3 = df_without_out_subj_cond_conc_RT
    elif rt_ot == "OT":
        df_rt_ot_1 = df_without_out_cond_conc_OT
        df_rt_ot_2 = df_without_out_subj_conc_OT
        df_rt_ot_3 = df_without_out_subj_cond_conc_OT
        
    plt.figure()
    pl_cond_after = sns.boxplot(x = "condition", y = rt_ot, data = df_rt_ot_1) #condition
    plt.figure()
    pl_subj_after = sns.boxplot(x = "Id", y = rt_ot, data = df_rt_ot_2) #subject

    
    #this loop creates a plot for each condition x id
    for cond in df_rt_ot_3["condition"].unique():  
        rddf_sub_after = df_rt_ot_3[df_rt_ot_3.condition == cond]
        plt.figure() #this creates a new figure on which your plot will appear
        pl_subj_cond_after = sns.boxplot(x = "Id", y = rt_ot, data = rddf_sub_after) # subject x condition
        pl_subj_cond_after.set_title(cond)

#########################    
#### Analysis for RT ####
#########################    

# based on https://link.springer.com/article/10.3758/s13428-011-0172-y/tables/2

#get the number of subjects
subjects = result_total['Id'].nunique()

#ONE-WAY REPEATED MEASURES ANOVA
#Print results with relevant SSs MS df F p

for diffdf in ["condRT", "scRT"]:
    
    if diffdf == "condRT":
        df = df_without_out_cond_conc_RT
        df = df.groupby(['Id', 'condition'], as_index=False)['RT'].mean()
        
    elif diffdf == "scRT":
        df = df_without_out_subj_cond_conc_RT
        df = df.groupby(['Id', 'condition'], as_index=False)['RT'].mean()
   
    
    lmfit = ols('RT ~ C(condition)+C(Id)',data=df).fit()
    results_summary_1 = sm.stats.anova_lm(lmfit, typ = 2)
    df_means = df.groupby(["condition"])['RT'].agg(['mean', 'std']) #calculate means

    ms_participant = (results_summary_1.iat[1,0])/(results_summary_1.iat[1,1])                                                    
    ms_cond = (results_summary_1.iat[0,0])/(results_summary_1.iat[0,1])                                                    
    ms_participant_x_cond = (results_summary_1.iat[2,0])/(results_summary_1.iat[2,1]) 
    results_summary_1.insert(2, "MS", [ms_cond, ms_participant, ms_participant_x_cond], True) 
    results_summary_1.loc["C(Id)",'F'] = ""
    results_summary_1.loc["Residual",'F'] = ""
    
    ss_cond = results_summary_1.iat[0,0]
    ss_error = results_summary_1.iat[2,0]
    results_summary_1['η2p'] = (ss_cond)/(ss_cond + ss_error), "", ""
    
    #now the fitted data 
    #convert Cond to numeric 
    
    def f(row):
        if row['condition'] == "spec":
            val = 1
        elif row['condition'] == "subrule":
            val = 2
        elif row ['condition'] == "rule":
            val = 3
        else:
            val = 4
        return val
            
    df['Cond_L'] = df.apply(f, axis=1)
    
    #Another 1-way ANOVA
    lmfit = ols('RT ~ Cond_L*C(Id)',data=df).fit()                                                   
    results_summary_2 = sm.stats.anova_lm(lmfit, typ = 2)
    results_summary_2 = results_summary_2.drop(['C(Id)', 'Residual'])
    #get SS Linear 
    ss_linear = results_summary_2.iat[0,0]
    ss_linear_x_participant = results_summary_2.iat[1,0]
    
    p_e_sq_2 = ss_linear/(ss_linear_x_participant + ss_linear)
               
    ms_linear_x_participant = ss_linear_x_participant/(subjects-1)                                                    
    F = (ss_linear/ ms_linear_x_participant)
    
    results_summary_2.insert(2, "MS", [ss_linear, ms_linear_x_participant], True) 
    results_summary_2.at["Cond_L","F"] = F
    results_summary_2.loc["Cond_L:C(Id)",'F'] = ""
    
    results_summary_2['η2p'] = p_e_sq_2, ""
    
    frames = [results_summary_1, results_summary_2]
    result = pd.concat(frames)
    #result = result.drop(['PR(>F)'], axis=1)
    
    #Polynomial Coefficients for 4 variables, linear = -3, 1, 1 ,3
    #check Statistical Methods for Psychology by Howell 
    slope = ss_linear/(8*((3*3)+(1*1)+(1*1)+(3*3)))
    slope = slope**(1/2)
    

    print("\n" + diffdf + "\n", result, "\n\nslope is:", slope, "\n__________________________________\n")
    
    #sns.regplot(x='Cond_L', y='RT', data=dataset)
    plt.figure()
    ax = sns.regplot(x='Cond_L', y='RT', data=df, x_estimator=np.mean)
    ax.set_title(diffdf)
    ax.set(ylabel='Reaction Times (RT)', xlabel='Distance')
    ax.set(xticks=np.arange(1, 5, 1))  #limit the number of ticks to 4
    ax.set_xticklabels(['spec','sub','rule', 'gen'])
    
    #plt.savefig('saving-a-seaborn-plot-as-pdf-file-300dpi.pdf', dpi = 300)
   
    
    ########################
    ###perform post-hocs###
    #######################

    # perform multiple pairwise comparison 
    t_test_PP = pg.pairwise_ttests(dv='RT', between='condition', data=df)#.round(3)
    print(t_test_PP)
    
    #check for a normal distribution
    s_array = df[["RT"]].to_numpy()
    shapiro_test, p_shapiro = stats.shapiro(s_array)
    print("\nshapiro test results:", shapiro_test, ",", p_shapiro)
    
    if p_shapiro > 0.05:
        print("p>0.05: normal distribution\n__________________________________\n__________________________________")
    else:
        print("not normal distribution\n__________________________________\n__________________________________")

#########################    
#### Analysis for OT ####
#########################    

#ONE-WAY REPEATED MEASURES ANOVA
#Print results with relevant SSs MS df F p

for diffdf in ["condOT", "scOT"]:
    if diffdf == "condOT":
        df = df_without_out_cond_conc_OT
        df = df.groupby(['Id', 'condition'], as_index=False)['OT'].mean()

    elif diffdf == "scOT":
        df = df_without_out_subj_cond_conc_OT
        df = df.groupby(['Id', 'condition'], as_index=False)['OT'].mean()

    lmfit = ols('OT ~ C(condition)+C(Id)',data=df).fit()
    results_summary_1 = sm.stats.anova_lm(lmfit, typ = 2)
    df_means = df.groupby(["condition"])['OT'].agg(['mean', 'std']) #calculate means
    
    ms_participant = (results_summary_1.iat[1,0])/(results_summary_1.iat[1,1])                                                    
    ms_cond = (results_summary_1.iat[0,0])/(results_summary_1.iat[0,1])                                                    
    ms_participant_x_cond = (results_summary_1.iat[2,0])/(results_summary_1.iat[2,1]) 
    results_summary_1.insert(2, "MS", [ms_cond, ms_participant, ms_participant_x_cond], True) 
    results_summary_1.loc["C(Id)",'F'] = ""
    results_summary_1.loc["Residual",'F'] = ""
    
    ss_cond = results_summary_1.iat[0,0]
    ss_error = results_summary_1.iat[2,0]
    results_summary_1['η2p'] = (ss_cond)/(ss_cond + ss_error), "", ""
    
    #now the fitted data 
    #convert Cond to numeric 
    
    def f(row):
        if row['condition'] == "spec":
            val = 1
        elif row['condition'] == "subrule":
            val = 2
        elif row ['condition'] == "rule":
            val = 3
        else:
            val = 4
        return val
            
    df['Cond_L'] = df.apply(f, axis=1)
    
    #Another 1-way ANOVA
    #to cover information missing from the first procedure
    lmfit = ols('OT ~ Cond_L*C(Id)',data=df).fit()                                                   
    results_summary_2 = sm.stats.anova_lm(lmfit, typ = 2)
    results_summary_2 = results_summary_2.drop(['C(Id)', 'Residual'])
    
    #get SS Linear 
    ss_linear = results_summary_2.iat[0,0]
    ss_linear_x_participant = results_summary_2.iat[1,0]
    
    p_e_sq_2 = ss_linear/(ss_linear_x_participant + ss_linear)
               
    ms_linear_x_participant = ss_linear_x_participant/(subjects-1)                                                    
    F = (ss_linear/ ms_linear_x_participant)
    
    results_summary_2.insert(2, "MS", [ss_linear, ms_linear_x_participant], True) 
    results_summary_2.at["Cond_L","F"] = F
    results_summary_2.loc["Cond_L:C(Id)",'F'] = ""
    
    results_summary_2['η2p'] = p_e_sq_2, ""
    
    frames = [results_summary_1, results_summary_2]
    result = pd.concat(frames)
    #result = result.drop(['PR(>F)'], axis=1)
    
    #Polynomial Coefficients for 4 variables, linear = -3, 1, 1 ,3
    #check Statistical Methods for Psychology by Howell 
    slope = ss_linear/(8*((3*3)+(1*1)+(1*1)+(3*3)))
    slope = slope**(1/2)
 
    print("\n" + diffdf + "\n", result, "\n\nslope is:", slope, "\n__________________________________\n")

    #sns.regplot(x='Cond_L', y='OT', data=dataset)
    plt.figure()
    ax = sns.regplot(x='Cond_L', y='OT', data=df, x_estimator=np.mean)
    ax.set_title(diffdf)
    ax.set(ylabel='Onset Times (OT)', xlabel='Distance')
    ax.set(xticks=np.arange(1, 5, 1))  #limit the number of ticks to 4
    ax.set_xticklabels(['spec','sub','rule', 'gen'])
    
    #save plots
    #plt.savefig('saving-a-seaborn-plot-as-pdf-file-300dpi.pdf', dpi = 300)       

    ########################
    ### perform post-hocs ###
    #######################
    
    # perform multiple pairwise comparison
    t_test_PP = pg.pairwise_ttests(dv='OT', between='condition', data=df).round(3)
    print(t_test_PP)
    
    #check for a normal distribution
    s_array = df[["OT"]].to_numpy()
    shapiro_test, p_shapiro = stats.shapiro(s_array)
    print("\nshapiro test results:", shapiro_test, ",", p_shapiro)
    
    if p_shapiro > 0.05:
        print("p>0.05: normal distribution\n__________________________________\n__________________________________")
    else:
        print("not normal distribution\n__________________________________\n__________________________________")


