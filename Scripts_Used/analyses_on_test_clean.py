#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:12:56 2020

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
from sklearn import preprocessing
from scipy.stats import normaltest
from statsmodels.formula.api import ols
import warnings

#avoid warning
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")
#for percentange
def percentage(percent, whole):
  return (percent * whole) / 100.0

# this script goes into the data training folder
data_folder_test = "/data/pt_02312/MRep_test_backup/data/"
folder_results_test = "/data/pt_02312/results_test/"

#create two empty dataframe where to store the results
df_result_te = pd.DataFrame([])

#read all the files in a folders and creat a list
filenames_te = os.listdir(data_folder_test)

#create a list only with files ending with the word "UPDATED"
#first create an empy list for Training and Test
UPDATED_files_te = []

#append the filenames to the lists
for filename_te in filenames_te:
    if filename_te.endswith("UPDATED.csv"):
        UPDATED_files_te.append(filename_te)

#sort the filenames in some order
UPDATED_files_te = sorted(UPDATED_files_te)

df_te_tot = []

# read the files in the test folder    
pattern = re.compile(r'(\d*)_Mental')
participant_numbers =  ["; ".join(pattern.findall(item)) for item in UPDATED_files_te]

for UPDATED_file_te in UPDATED_files_te: #UPDATED_file_te = each participant (number)
    
    #print(UPDATED_file_te)
    
    participant_number_te = [int(s) for s in re.findall(r'(\d*)_Mental', UPDATED_file_te)]
    str_number_te = ' '.join(map(str, participant_number_te))
    
    # with fillers
    df_te = pd.read_csv(data_folder_test + UPDATED_file_te,  header=0)
    
    df_te = df_te.loc[df_te['file_n'] >= 1]

    #check if someone is below 90% accuracy
    count_row = df_te['resp_total_corr'].shape[0] 
    score_total = df_te['resp_total_corr'].sum()
    
    if score_total < round(percentage(90, count_row)): 
        print(participant_number_te + "is below 90% accuracy")   

    df_te = df_te.loc[(df_te['trial_type'] == "experimental")]
    df_te = df_te.loc[(df_te['resp_total_corr'] == 1)]

    file_values = df_te.file_n.unique() 
    
    df_te_tot.append(df_te)

df_te_tot = pd.concat(df_te_tot)
df_te_tot = df_te_tot[["participant", "resp1.rt", "resp_total_time", "conditions"]]
df_te_tot['ET'] = df_te_tot['resp_total_time'] - df_te_tot['resp1.rt']  
df_te_tot.to_csv(folder_results_test + "results_test_with_outliers.csv", index = False, header=True)



############################
############################
#####ANALYSES ON TEST#######
############################
############################

result_total = df_te_tot

result_total['conditions'].replace(["spec"], '1_specific', inplace=True)
result_total['conditions'].replace(["subrule"], '2_subrule', inplace=True)
result_total['conditions'].replace(["rule"], '3_rule', inplace=True)
result_total['conditions'].replace(["general"], '4_general', inplace=True)

result_total = result_total.sort_values('conditions')

result_total['conditions'].replace(["1_specific"], 'spec', inplace=True)
result_total['conditions'].replace(["2_subrule"], 'subrule', inplace=True)
result_total['conditions'].replace(["3_rule"], 'rule', inplace=True)
result_total['conditions'].replace(["4_general"], 'general', inplace=True)

result_total = result_total.rename(columns = {'participant': 'Id', 'resp1.rt': 'OT', 
                                              'resp_total_time': 'RT', 'conditions': 'condition'}, inplace = False)

rddf_OT = result_total[['Id','condition', "OT"]]
rddf_RT = result_total[['Id','condition', "RT"]]
rddf_ET = result_total[['Id','condition', "ET"]]

#### identifying outliers ####
## boxplots
#for rt_ot in ["RT","OT"]:
#    
#    if rt_ot == "RT":
#        df_rt_ot = rddf_RT
#    elif rt_ot == "OT":
#        df_rt_ot = rddf_OT
#        
#    pl_cond = sns.boxplot(x = "condition", y = rt_ot, data = df_rt_ot) #condition
#    plt.figure()
#    pl_subj = sns.boxplot(x = "Id", y = rt_ot, data = df_rt_ot) #subject
#
#    #this loop creates a plot for each condition x id
#    for cond in df_rt_ot["condition"].unique():  
#        rddf_sub = df_rt_ot[df_rt_ot.condition == cond]
#        plt.figure() #this creates a new figure on which your plot will appear
#        pl_subj_cond = sns.boxplot(x = "Id", y = rt_ot, data = rddf_sub) # subject x condition
#        pl_subj_cond.set_title(cond)
#        plt.figure()
        
## IQR 
#create empty dataframes for appending and concatenating 
df_without_out_cond_conc_RT = [] 
df_only_out_cond_conc_RT = [] 
df_without_out_cond_conc_OT= [] 
df_only_out_cond_conc_OT = [] 
df_without_out_cond_conc_ET= [] 
df_only_out_cond_conc_ET = [] 




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

for rt_ot in ["RT","OT", "ET"]:
   
    if rt_ot == "RT":
        df_rt_ot = rddf_RT
        df_1 = df_without_out_cond_conc_RT
        df_2 = df_only_out_cond_conc_RT
    elif rt_ot == "OT":
        df_rt_ot = rddf_OT
        df_1 = df_without_out_cond_conc_OT
        df_2 = df_only_out_cond_conc_OT
    elif rt_ot == "ET":
        df_rt_ot = rddf_ET
        df_1 = df_without_out_cond_conc_ET
        df_2 = df_only_out_cond_conc_ET
        
    #CONDITIONS
    for cond in df_rt_ot["condition"].unique():
        outliers_focus = "condition"
        rddf_sub_out = df_rt_ot[df_rt_ot.condition == cond]
        df_without_out_cond, df_only_out_cond = iqr_func(rddf_sub_out) #extract the variables form the function

        df_1.append(df_without_out_cond) #append condition dataframes after each loop
        df_2.append(df_only_out_cond) 

def conc(output_df):
    output_df = pd.concat(output_df)
    output_df = output_df.reset_index(drop=True)   

    return output_df


df_without_out_cond_conc_RT, df_only_out_cond_conc_RT = (conc(df) for df in [df_without_out_cond_conc_RT, df_only_out_cond_conc_RT]) # or either .map or .applymap
df_without_out_cond_conc_OT, df_only_out_cond_conc_OT = (conc(df) for df in [df_without_out_cond_conc_OT, df_only_out_cond_conc_OT])
df_without_out_cond_conc_ET, df_only_out_cond_conc_ET = (conc(df) for df in [df_without_out_cond_conc_ET, df_only_out_cond_conc_ET])

# boxplots to check results
#for rt_ot in ["RT","OT", "ET"]:
#    
#    if rt_ot == "RT":
#        df_rt_ot_1 = df_without_out_cond_conc_RT
#    elif rt_ot == "OT":
#        df_rt_ot_1 = df_without_out_cond_conc_OT
#    elif rt_ot == "ET":
#        df_rt_ot_1 = df_without_out_cond_conc_ET       
#    plt.figure()
#    pl_cond_after = sns.boxplot(x = "condition", y = rt_ot, data = df_rt_ot_1) #condition
#    plt.figure()
#    pl_subj_after = sns.boxplot(x = "Id", y = rt_ot, data = df_rt_ot_2) #subject

    

#########################    
#### Analysis for RT ####
#########################    

# based on https://link.springer.com/article/10.3758/s13428-011-0172-y/tables/2

#get the number of subjects
subjects = result_total['Id'].nunique()

#ONE-WAY REPEATED MEASURES ANOVA
#Print results with relevant SSs MS df F p

for diffdf in ["condRT","condOT", "condET"]:
    
    if diffdf == "condRT":
        df = df_without_out_cond_conc_RT
        df = df.groupby(['Id', 'condition'], as_index=False)['RT'].mean()
        source = "RT"
        source_label = "Mean RT (s)"
    elif diffdf == "condOT":
        print("\n###################################\n")
        df = df_without_out_cond_conc_OT
        df = df.groupby(['Id', 'condition'], as_index=False)['OT'].mean()
        source = "OT" 
        source_label = "Mean OTs (s)"
    elif diffdf == "condET":
        print("\n###################################\n")
        df = df_without_out_cond_conc_ET
        df = df.groupby(['Id', 'condition'], as_index=False)['ET'].mean()
        source = "ET"  
        source_label = "Mean ETs (s)"
       
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
    df.to_csv(folder_results_test + source + ".csv", index=False)   

    df_condition = df.groupby(["condition"])[source].agg(['mean', 'std'])
    df_condition_subject = df.groupby(["condition", "Id"])[source].agg(['mean']).reset_index()
    df_cond_sub_gen = df_condition_subject.loc[(df_condition_subject['condition'] == "general")]
    df_cond_sub_gen.columns = ['condition', 'Id', "general"]
    df_cond_sub_gen = df_cond_sub_gen.reset_index(drop=True)
    df_cond_sub_gen.drop(['condition', "Id"], axis =1, inplace = True)
    df_cond_sub_srule = df_condition_subject.loc[(df_condition_subject['condition'] == "subrule")]
    df_cond_sub_srule.columns = ['condition', 'Id', "sub_rule"]    
    df_cond_sub_srule = df_cond_sub_srule.reset_index(drop=True)
    df_cond_sub_srule.drop(['condition', "Id"], axis =1, inplace = True)
    df_cond_sub_rule = df_condition_subject.loc[(df_condition_subject['condition'] == "rule")] 
    df_cond_sub_rule.columns = ['condition', 'Id', "rule"]
    df_cond_sub_rule.drop(['condition', "Id"], axis =1, inplace = True)
    df_cond_sub_rule = df_cond_sub_rule.reset_index(drop=True)
    df_cond_sub_spec = df_condition_subject.loc[(df_condition_subject['condition'] == "spec")]    
    df_cond_sub_spec.columns = ['condition', 'Id', "spec"]
    df_cond_sub_spec.drop(['condition'], axis =1, inplace = True)
    df_cond_sub_spec = df_cond_sub_spec.reset_index(drop=True)
    df_condition_subject = pd.concat([df_cond_sub_spec, df_cond_sub_rule, df_cond_sub_srule, df_cond_sub_gen], axis=1, sort=False)
    df_condition_subject.to_csv(folder_results_test + source + "_posthoc.csv", index=False)
    
    #check for a normal distribution
    s_array = df[[source]].to_numpy()
    stat, p_d_agostino = normaltest(s_array)
    print('Statistics=%.3f, p=%.3f' % (stat, p_d_agostino))
    
    if p_d_agostino > 0.05:
        print("p>0.05: normal distribution\n__________________________________\n__________________________________")
        normalized = s_array
    else:
        print("not normal distribution\n__________________________________\n__________________________________") 
        pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
        #other method = yeo-johnson - gives in output positive values
        normalized = pt.fit_transform(s_array)
        stat, p_d_agostino = normaltest(normalized)
        print('AFTER TRANSFORMATION: Statistics=%.3f, p=%.3f' % (stat, p_d_agostino))
        df[source] = normalized
        df.to_csv(r"/data/pt_02312/results_test/" + source + "_normalized.csv", index=False)   
        
        df_condition = df.groupby(["condition"])[source].agg(['mean', 'std']) 
        df_condition_subject = df.groupby(["condition", "Id"])[source].agg(['mean']).reset_index()
        df_cond_sub_gen = df_condition_subject.loc[(df_condition_subject['condition'] == "general")]
        df_cond_sub_gen.columns = ['condition', 'Id', "general"]
        df_cond_sub_gen = df_cond_sub_gen.reset_index(drop=True)
        df_cond_sub_gen.drop(['condition', "Id"], axis =1, inplace = True)
        df_cond_sub_srule = df_condition_subject.loc[(df_condition_subject['condition'] == "subrule")]
        df_cond_sub_srule.columns = ['condition', 'Id', "sub_rule"]    
        df_cond_sub_srule = df_cond_sub_srule.reset_index(drop=True)
        df_cond_sub_srule.drop(['condition', "Id"], axis =1, inplace = True)
        df_cond_sub_rule = df_condition_subject.loc[(df_condition_subject['condition'] == "rule")] 
        df_cond_sub_rule.columns = ['condition', 'Id', "rule"]
        df_cond_sub_rule.drop(['condition', "Id"], axis =1, inplace = True)
        df_cond_sub_rule = df_cond_sub_rule.reset_index(drop=True)
        df_cond_sub_spec = df_condition_subject.loc[(df_condition_subject['condition'] == "spec")]    
        df_cond_sub_spec.columns = ['condition', 'Id', "spec"]
        df_cond_sub_spec.drop(['condition'], axis =1, inplace = True)
        df_cond_sub_spec = df_cond_sub_spec.reset_index(drop=True)
        df_condition_subject = pd.concat([df_cond_sub_spec, df_cond_sub_rule, df_cond_sub_srule, df_cond_sub_gen], axis=1, sort=False)
        df_condition_subject.to_csv(folder_results_test+ source + "_norm_posthoc.csv", index=False)         
  
    sns.distplot(normalized, hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})    
    
    plt.savefig(folder_results_test + "plots/" + source + "_dist_plot.jpg")

    #1-way ANOVA for testing Linear Regression
    lmfit = ols(source + " ~ C(condition)+C(Id)",data=df).fit()
    results_summary_1 = sm.stats.anova_lm(lmfit, typ = 2)
    df_means = df.groupby(["condition"])[source].agg(['mean', 'std']) #calculate means

    ms_participant = (results_summary_1.iat[1,0])/(results_summary_1.iat[1,1])                                                    
    ms_cond = (results_summary_1.iat[0,0])/(results_summary_1.iat[0,1])                                                    
    ms_participant_x_cond = (results_summary_1.iat[2,0])/(results_summary_1.iat[2,1]) 
    results_summary_1.insert(2, "MS", [ms_cond, ms_participant, ms_participant_x_cond], True) 
    results_summary_1.loc["C(Id)",'F'] = ""
    results_summary_1.loc["Residual",'F'] = ""
    
    ss_cond = results_summary_1.iat[0,0]
    ss_error = results_summary_1.iat[2,0]
    results_summary_1['η2p'] = (ss_cond)/(ss_cond + ss_error), "", ""

    #ANOTHER 1-way ANOVA for testing Linear Regression
    lmfit = ols(source + " ~ Cond_L*C(Id)",data=df).fit()                                                   
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
    p_e_sq_2 = float("{:.4f}".format(p_e_sq_2))
    
    frames = [results_summary_1, results_summary_2]
    result = pd.concat(frames)
    p_value_ANOVA = results_summary_1.loc["C(condition)"]["PR(>F)"]
    #Polynomial Coefficients for 4 variables, linear = -3, 1, 1 ,3
    #check Statistical Methods for Psychology by Howell 
    slope = ss_linear/(8*((3*3)+(1*1)+(1*1)+(3*3)))
    slope = (slope**(1/2))
    slope = float("{:.4f}".format(slope))
    

    print("\n" + diffdf + "\n", result, "\n\nslope is:", slope, "\n__________________________________\n")
    
    plt.figure()
    ax = sns.regplot(x='Cond_L', y= source, data=df, x_estimator=np.mean)
    ax.set_title(diffdf)
    ax.set(ylabel= source_label, xlabel='condition')
    ax.set(xticks=np.arange(1, 5, 1))  #limit the number of ticks to 4
    ax.set_xticklabels(['Specific','Sub-Rule','Rule', 'General'])

    if source == "OT":    
        plt.text((2.5+3)*.5, 0.39, "slope = " + str(slope), ha='left', va='center', color="black", fontsize=10)
        plt.text((2.5+3)*.5, 0.42, "η2p = " + str(p_e_sq_2), ha='left', va='center', color="black", fontsize=10)
    
    plt.savefig(folder_results_test + "plots/" + source + "_reg_plot.png", dpi = 300)
    
    plt.figure()
    res = sns.residplot('Cond_L',source,data=df) 
    res.set_xticklabels(["", 'Specific',"", 'Sub-Rule', "", 'Rule', "", 'General'])
    plt.savefig(folder_results_test + "plots/" + source + "_resid_plot.png", dpi=300)
    plt.figure()

    ########################
    #####ONE-WAY ANOVA#####
    #######################
    
    ANOVA_1W = ols(source + " ~ C(condition)", data= df).fit()
    ANOVA_1W_table = sm.stats.anova_lm(ANOVA_1W, typ=2)
    
    ########################
    ###perform post-hocs###
    #######################

    # perform multiple pairwise comparison 
    t_test_PP = pg.pairwise_ttests(dv=source, within='condition', subject='Id', data=df)
    t_test_df = t_test_PP.round(3)
    t_test_df = t_test_df.drop(['Contrast', 'BF10', "Parametric", "Paired", "hedges"], axis = 1)
    t_test_df["p-unc"] = 1.5 * t_test_df["p-unc"]
    t_test_df['Tail'] = t_test_df['Tail'].str.replace('two','one')
    t_test_df = t_test_df.loc[((t_test_df['A'] == "general") & (t_test_df['B'] == "rule") 
    | (t_test_df['A'] == "rule") & (t_test_df['B'] == "subrule") 
    | (t_test_df['A'] == "spec") & (t_test_df['B'] == "subrule"))]
    t_test_df.reset_index(inplace=True, drop=True)

    plt.figure()
    my_pal = {"spec": "m", "subrule":"whitesmoke", "rule":"grey", "general":"black"}
    box_post_hoc = sns.boxplot(x = "condition", y = source, data = df, order=["spec", "subrule", "rule", "general"],palette=my_pal) 
    box_post_hoc.set(ylabel= source_label, xlabel='condition')
    box_post_hoc.set_xticklabels(['Specific','Sub-Rule','Rule', 'General'])
    for patch in box_post_hoc.artists:
     r, g, b, a = patch.get_facecolor()
     patch.set_facecolor((r, g, b, .8))
#https://python-graph-gallery.com/33-control-colors-of-boxplot-seaborn/
    
    # statistical annotation
    if source == "OT":
        x1, x2 = 2, 3  
        y, h, col = df[source].max() + .08, .08, 'k'
        plt.plot([x1, x1, x2, x2], [1, 1.04, 1.04, 1], lw=1.2, c=col)
        plt.text((x1+x2)*.5, 1.05, "***", ha='center', va='center', color=col, fontsize=12)
        x2, x3 = 1, 2  
        y, h, col = df[source].max() + .04, .04, 'k'
        plt.plot([x2, x2, x3, x3], [0.9, 0.94, 0.94, 0.9], lw=1.2, c=col)
        plt.text((x2+x3)*.5, .95, "***", ha='center', va='center', color=col,  fontsize=12)
        x3, x4 = 0, 1  
        y, h, col = df[source].max() + .02, .02, 'k'
        plt.plot([x3, x3, x4, x4], [0.8, .84, .84, .8], lw=1.2, c=col)
        plt.text((x3+x4)*.5, .85, "**", ha='center', va='center', color=col,  fontsize=12)
    
    
    plt.savefig(folder_results_test + "plots/" + source + "_boxplot.png", dpi = 300)
    plt.figure()