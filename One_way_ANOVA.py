#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:04:12 2020

@author: papitto
"""

#### ONE WAY ANOVA ####
# Perform a one way repeated measures

#pip install pandas statsmodels

import pandas as pd
from statsmodels.stats.anova import AnovaRM

input_file = '/home/raid2/papitto/Desktop/PsychoPy/MRep_2020_backup/MRep_training_backup/data/results.csv'
dataset = pd.read_csv(input_file)

dataset = dataset[["Subj_tr", "OT3_spec_Tot", "OT3_sub_Tot", "OT3_rule_Tot", "OT3_gen_Tot", "RT3_spec_Tot", "RT3_sub_Tot", "RT3_rule_Tot", "RT3_gen_Tot" ]]

OT_gen = dataset["OT3_gen_Tot"]
OT_spec = dataset["OT3_spec_Tot"]
OT_sub = dataset["OT3_sub_Tot"]
OT_rule = dataset["OT3_rule_Tot"]

dataset_spec = dataset[["Subj_tr", "OT3_spec_Tot", "RT3_spec_Tot"]]
dataset_spec = dataset_spec.rename(columns={"OT3_spec_Tot": "OT","RT3_spec_Tot": "RT" })
dataset_sub = dataset[["Subj_tr", "OT3_sub_Tot", "RT3_sub_Tot"]]
dataset_sub = dataset_sub.rename(columns={"OT3_sub_Tot": "OT", "RT3_sub_Tot": "RT"})
dataset_rule = dataset[["Subj_tr", "OT3_rule_Tot", "RT3_rule_Tot"]]
dataset_rule = dataset_rule.rename(columns={"OT3_rule_Tot": "OT","RT3_rule_Tot": "RT"})
dataset_gen = dataset[["Subj_tr", "OT3_gen_Tot", "RT3_gen_Tot"]]
dataset_gen = dataset_gen.rename(columns={"OT3_gen_Tot": "OT","RT3_gen_Tot": "RT"})


for i, row in dataset_spec.iterrows():
    dataset_spec.at[i, 'condition'] = "spec"
for i, row in dataset_sub.iterrows():
    dataset_sub.at[i, 'condition'] = "sub"
for i, row in dataset_rule.iterrows():
    dataset_rule.at[i, 'condition'] = "rule"
for i, row in dataset_gen.iterrows():
    dataset_gen.at[i, 'condition'] = "gen"

frames = [dataset_spec, dataset_sub, dataset_rule, dataset_gen] 
result_df = pd.concat(frames)   

aovrm = AnovaRM(result_df, 'OT', 'Sub_tr', within=['cond'])
res = aovrm.fit()

print(res) 
