#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:04:12 2020

@author: papitto
"""

#### ONE WAY ANOVA ####
# Perform a one way repeated measures

# if you dont have the stastsmodels package just install it from here:
#pip install pandas statsmodels

import pandas as pd
from statsmodels.stats.anova import AnovaRM

# this reads the result file created from the previous script
input_file = '/home/raid2/papitto/Desktop/PsychoPy/MRep_2020_backup/MRep_training_backup/data/results.csv'
dataset = pd.read_csv(input_file)

#only select the necessary columns
dataset = dataset[["Subj_tr", "OT3_spec_Tot", "OT3_sub_Tot", "OT3_rule_Tot", "OT3_gen_Tot", "RT3_spec_Tot", "RT3_sub_Tot", "RT3_rule_Tot", "RT3_gen_Tot" ]]

#create a dataset for each condition
dataset_spec = dataset[["Subj_tr", "OT3_spec_Tot", "RT3_spec_Tot"]]
dataset_spec = dataset_spec.rename(columns={"OT3_spec_Tot": "OT","RT3_spec_Tot": "RT" })
dataset_sub = dataset[["Subj_tr", "OT3_sub_Tot", "RT3_sub_Tot"]]
dataset_sub = dataset_sub.rename(columns={"OT3_sub_Tot": "OT", "RT3_sub_Tot": "RT"})
dataset_rule = dataset[["Subj_tr", "OT3_rule_Tot", "RT3_rule_Tot"]]
dataset_rule = dataset_rule.rename(columns={"OT3_rule_Tot": "OT","RT3_rule_Tot": "RT"})
dataset_gen = dataset[["Subj_tr", "OT3_gen_Tot", "RT3_gen_Tot"]]
dataset_gen = dataset_gen.rename(columns={"OT3_gen_Tot": "OT","RT3_gen_Tot": "RT"})

#add a condition column to each dataframe
for i, row in dataset_spec.iterrows():
    dataset_spec.at[i, 'condition'] = "spec"
for i, row in dataset_sub.iterrows():
    dataset_sub.at[i, 'condition'] = "sub"
for i, row in dataset_rule.iterrows():
    dataset_rule.at[i, 'condition'] = "rule"
for i, row in dataset_gen.iterrows():
    dataset_gen.at[i, 'condition'] = "gen"

# concatenate all the dataframes
frames = [dataset_spec, dataset_sub, dataset_rule, dataset_gen] 
result_df = pd.concat(frames)   

#perform the ANOVA
aovrm = AnovaRM(result_df, 'OT', 'Subj_tr', within=['condition'])
res = aovrm.fit()

print(res) 


################
##### use ######
### pingouin ###
################

import pingouin as pg
from pingouin import mixed_anova, read_dataset

df_ANOVA = result_df.rm_anova(dv='OT', within='condition', subject='Subj_tr', detailed=True)
df_means = result_df.groupby(["condition"])['OT'].agg(['mean', 'std']).round(2) #means + SD

################
##### plot #####
################

import seaborn as sns
sns.set(style="darkgrid")
tips = sns.load_dataset("tips")
ax = sns.pointplot(data=result_df, x='condition', y='OT', capsize=.2, order=["spec", "sub", "rule", "gen"])
ax.set(ylabel='Onset Time (OT)', xlabel='Condition')

################
#### p-hoc #####
################

# FDR-corrected post hocs with Hedges'g effect size
df_post_hoc = result_df.pairwise_ttests(dv='OT', within='condition', subject='Subj_tr',
                             parametric=True, padjust='fdr_bh', effsize='hedges')

