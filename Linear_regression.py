#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 11:43:23 2020

@author: papitto
"""
from statsmodels.formula.api import ols
import pandas as pd  

input_file = '/home/raid2/papitto/Desktop/PsychoPy/MRep_2020_backup/MRep_training_backup/data/results.csv'
dataset = pd.read_csv(input_file)

dataset = dataset[["Subj_tr", "OT3_spec_Tot", "OT3_sub_Tot", "OT3_rule_Tot", "OT3_gen_Tot" ]]


OT_gen = dataset["OT3_gen_Tot"]
OT_spec = dataset["OT3_spec_Tot"]
OT_sub = dataset["OT3_sub_Tot"]
OT_rule = dataset["OT3_rule_Tot"]

dataset_spec = dataset[["Subj_tr", "OT3_spec_Tot"]]
dataset_spec = dataset_spec.rename(columns={"OT3_spec_Tot": "OT"})
dataset_sub = dataset[["Subj_tr", "OT3_sub_Tot"]]
dataset_sub = dataset_sub.rename(columns={"OT3_sub_Tot": "OT"})
dataset_rule = dataset[["Subj_tr", "OT3_rule_Tot"]]
dataset_rule = dataset_rule.rename(columns={"OT3_rule_Tot": "OT"})
dataset_gen = dataset[["Subj_tr", "OT3_gen_Tot"]]
dataset_gen = dataset_gen.rename(columns={"OT3_gen_Tot": "OT"})


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

result_df['Spec']=result_df.condition.map({'spec':1,'sub':0,'rule':0, "gen": 0})
result_df['Sub']=result_df.condition.map({'sub':1,'spec':0,'rule':0, "gen": 0})
result_df['Rule']=result_df.condition.map({'rule':1,'spec':0,'sub':0, "gen": 0})
result_df['Gen']=result_df.condition.map({'gen':1,'spec':0,'rule':0, "sub": 0})


fit = ols('OT ~ C(Spec) + C(Sub) + C(Rule)+ C(Gen)', data=result_df).fit() 

fit.summary()