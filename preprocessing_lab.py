# this script takes as input files the results files (UPDATED ones) and create a unique row for each participant
# each row contains all the information concerning Training 1 (Tr1), Training 2 (Tr2) and Test (Te)

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
    
# this script goes into the data training folder
#change the position of the folders according to where you have the data
#data_folder_training = "/home/raid2/papitto/Desktop/PsychoPy/MRep_2020_backup/MRep_training_backup/data/"
#data_folder_test = "/home/raid2/papitto/Desktop/PsychoPy/MRep_2020_backup/MRep_test_backup/data/"
data_folder_training ="/data/pt_02312/MRep_training_backup/data/"
data_folder_test = "/data/pt_02312/MRep_test_backup/data/"



#create two empty dataframe where to store the results
df_result_tr = pd.DataFrame([]) #empty dataframe for result file
df_result_te = pd.DataFrame([])

#read all the files in a folders and creat a list
filenames_tr = os.listdir(data_folder_training)
filenames_te = os.listdir(data_folder_test)

#create a list only with files ending with the word "UPDATED"
#first create an empy list for Training and Test
UPDATED_files_tr = []
UPDATED_files_te = []

#append the filenames to the lists
for filename_tr in filenames_tr:
    if filename_tr.endswith("UPDATED.csv"):
        UPDATED_files_tr.append(filename_tr)
for filename_te in filenames_te:
    if filename_te.endswith("UPDATED.csv"):
        UPDATED_files_te.append(filename_te)

#sort the filenames in some order
UPDATED_files_tr = sorted(UPDATED_files_tr)
UPDATED_files_te = sorted(UPDATED_files_te)

#extract the number of the participants for Tr
pattern = re.compile(r'(\d*)_Mental')
participant_numbers =  ["; ".join(pattern.findall(item)) for item in UPDATED_files_tr]

###############################
######### TRAINING 1 #########
##############################

#read all the UPDATED files one at the time
# this will read automatically all the UPDATED files (keep the folder clean = remove files of excluded participants)
for UPDATED_file_tr in UPDATED_files_tr: 
    
    participant_number_tr = [int(s) for s in re.findall(r'(\d*)_Mental', UPDATED_file_tr)]
    str_number = ' '.join(map(str, participant_number_tr)) 

    df_tr = pd.read_csv(data_folder_training + UPDATED_file_tr,  header=0) #read the data file in
    
    #only keep rows referring to trials (experimental or filler)
    df_exp_fil_trials = df_tr.loc[(df_tr['trial_type'] == "experimental")|(df_tr['trial_type'] == "filler")]
    
    group = df_exp_fil_trials["group"].iloc[0] #extract group information (A, B, C, or D)

    # ALL LOOPS TOGETHER
    #this includes filler trilas
    # Corr_R = times they replied correctly to "is cue 2 encoded by cue 1"
    # Corr_S = times they perfomed the tapping sequence correctly
    # "repeat_training_loop1.thisRepN" is the value assigned in the original file to training 1 loops
    df_training_1 = df_exp_fil_trials.loc[df_exp_fil_trials['repeat_training_loop1.thisRepN'] >= 0]
    Corr_R_Tot = df_training_1["resp_R.corr"].sum() #number of correct relationship answers - all loops
    Corr_S_Tot = df_training_1["resp_total_corr"].sum() #number of correct sequences - all loops
    
    #only experimental trials
    df_training_1_wo = df_exp_fil_trials.loc[(df_exp_fil_trials['repeat_training_loop1.thisRepN'] >= 0) & (df_exp_fil_trials['trial_type'] == "experimental")]
    Corr_R_Tot_wo = df_training_1_wo["resp_R.corr"].sum() #number of correct relationship answers - all loops
    Corr_S_Tot_wo = df_training_1_wo["resp_total_corr"].sum() #number of correct sequences - all loops
    
    #create empty dictionaries
    loop_n_0 = {}    
    loop_n_1 = {} 
    loop_n_2 = {}
    loop_n_3 = {} 
    loop_n_4 = {}
    
    #get information ofr each loop (without specifying the condition)
    #wo = without filler trials
    #iteration from(0,6) = the loop is repeated six times one for each experimental loop (our 38 trials)
    for iterations in range(0,6):
        df_training_1_loop = df_training_1.loc[df_training_1['repeat_training_loop1.thisRepN'] == iterations]
        Total_resp_R_loop = df_training_1_loop["resp_R.corr"].sum() #number of correct relationship answers (incl. fillers) - all loops
        Total_resp_Seq_loop = df_training_1_loop["resp_total_corr"].sum()
        loop_n_1[iterations] = df_training_1_loop["resp_R.corr"].sum() #loop with its response value
        loop_n_2[iterations] = df_training_1_loop["resp_total_corr"].sum()
        
        df_training_1_loop_wo = df_training_1_wo.loc[df_training_1_wo['repeat_training_loop1.thisRepN'] == iterations]
        Total_resp_R_loop_wo = df_training_1_loop_wo["resp_R.corr"].sum() #number of correct relationship answers (incl. fillers) - all loops
        Total_resp_Seq_loop_wo = df_training_1_loop_wo["resp_total_corr"].sum()
        loop_n_3[iterations] = df_training_1_loop_wo["resp_R.corr"].sum() #loop with its response value
        loop_n_4[iterations] = df_training_1_loop_wo["resp_total_corr"].sum()   
        
    df_Corr_R_per_loop = pd.DataFrame.from_dict(loop_n_1,  orient='index')
    df_Corr_R_per_loop = df_Corr_R_per_loop.transpose()
    df_Corr_R_per_loop.columns = ["Corr_R_1", "Corr_R_2", "Corr_R_3", "Corr_R_4", "Corr_R_5", "Corr_R_6"]
    
    df_Corr_R_per_loop_wo = pd.DataFrame.from_dict(loop_n_3,  orient='index')
    df_Corr_R_per_loop_wo = df_Corr_R_per_loop_wo.transpose()   
    df_Corr_R_per_loop_wo.columns = ["Corr_R_1_wo", "Corr_R_2_wo", "Corr_R_3_wo", "Corr_R_4_wo", "Corr_R_5_wo", "Corr_R_6_wo"]
    
    df_Corr_Seq1_per_loop = pd.DataFrame.from_dict(loop_n_2,  orient='index')
    df_Corr_Seq1_per_loop = df_Corr_Seq1_per_loop.transpose()
    df_Corr_Seq1_per_loop.columns = ["Corr_S1_1", "Corr_S1_2", "Corr_S1_3", "Corr_S1_4", "Corr_S1_5", "Corr_S1_6"]  
    
    df_Corr_Seq1_per_loop_wo = pd.DataFrame.from_dict(loop_n_4,  orient='index')
    df_Corr_Seq1_per_loop_wo = df_Corr_Seq1_per_loop_wo.transpose()   
    df_Corr_Seq1_per_loop_wo.columns = ["Corr_S1_1_wo", "Corr_S1_2_wo", "Corr_S1_3_wo", "Corr_S1_4_wo", "Corr_S1_5_wo", "Corr_S1_6_wo"]
    
    #empty the previous dictionaries
    loop_n_1.clear() 
    loop_n_2.clear()
    loop_n_1.clear()
    loop_n_2.clear()
    
    # Corr_R for each loop TR1 (specifying the experimentla condition)
    # FILLER TRIALS ARE NOT CONSIDERED ANYMORE
    for iterations in range(0,6):
        df_training_1_loop = df_training_1.loc[(df_training_1['repeat_training_loop1.thisRepN'] == iterations) & (df_training_1['resp_total_corr'] == 1)]
        df_tr_1_spec_loop = df_training_1_loop.loc[(df_training_1_loop['conditions'] == "spec")]
        df_tr_1_sub_loop = df_training_1_loop.loc[(df_training_1_loop['conditions'] == "subrule")]
        df_tr_1_rule_loop = df_training_1_loop.loc[(df_training_1_loop['conditions'] == "rule")]
        df_tr_1_gen_loop = df_training_1_loop.loc[(df_training_1_loop['conditions'] == "general")]
        
        Corr_R_Spec_loop_Tr1 = df_tr_1_spec_loop["resp_R.corr"].sum()
        Corr_R_Sub_loop_Tr1 = df_tr_1_sub_loop["resp_R.corr"].sum()
        Corr_R_Rule_loop_Tr1 = df_tr_1_rule_loop["resp_R.corr"].sum()
        Corr_R_Gen_loop_Tr1 = df_tr_1_gen_loop["resp_R.corr"].sum()      
        
        #assign each loop_n dictionary to a condition 
        loop_n_1[iterations] = df_tr_1_spec_loop["resp_R.corr"].sum() #loop with its response value
        loop_n_2[iterations] = df_tr_1_sub_loop["resp_R.corr"].sum()
        loop_n_3[iterations] = df_tr_1_rule_loop["resp_R.corr"].sum()
        loop_n_4[iterations] = df_tr_1_gen_loop["resp_R.corr"].sum()
         
    #create  a dataframe from the previous 4 dictionaries (one per condition)           
    df_Corr_R_spec_per_loop = pd.DataFrame.from_dict(loop_n_1,  orient='index')
    df_Corr_R_spec_per_loop = df_Corr_R_spec_per_loop.transpose() #invert x and y axis
    df_Corr_R_spec_per_loop.columns = ["Corr_R_spec_1", "Corr_R_spec_2", "Corr_R_spec_3", "Corr_R_spec_4", "Corr_R_spec_5", "Corr_R_spec_6"]
    
    df_Corr_R_sub_per_loop = pd.DataFrame.from_dict(loop_n_2,  orient='index')
    df_Corr_R_sub_per_loop = df_Corr_R_sub_per_loop.transpose()
    df_Corr_R_sub_per_loop.columns = ["Corr_R_sub_1", "Corr_R_sub_2", "Corr_R_sub_3", "Corr_R_sub_4", "Corr_R_sub_5", "Corr_R_sub_6"]
    
    df_Corr_R_rule_per_loop = pd.DataFrame.from_dict(loop_n_3,  orient='index')
    df_Corr_R_rule_per_loop = df_Corr_R_rule_per_loop.transpose()
    df_Corr_R_rule_per_loop.columns = ["Corr_R_rule_1", "Corr_R_rule_2", "Corr_R_rule_3", "Corr_R_rule_4", "Corr_R_rule_5", "Corr_R_rule_6"]
    
    df_Corr_R_gen_per_loop = pd.DataFrame.from_dict(loop_n_4,  orient='index')
    df_Corr_R_gen_per_loop = df_Corr_R_gen_per_loop.transpose()
    df_Corr_R_gen_per_loop.columns = ["Corr_R_gen_1", "Corr_R_gen_2", "Corr_R_gen_3", "Corr_R_gen_4", "Corr_R_gen_5", "Corr_R_gen_6"]
    
    # Corr_R for each loop TR1 (specifying the experimentla condition)    
    for iterations in range(0,6):
        df_training_1_loop = df_training_1.loc[(df_training_1['repeat_training_loop1.thisRepN'] == iterations) & (df_training_1['resp_total_corr'] == 1)]
        df_tr_1_spec_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "spec")] #& (df_training_1['did_get_here'] == 1)]
        df_tr_1_sub_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "subrule")]
        df_tr_1_rule_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "rule")]
        df_tr_1_gen_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "general")]
            
        Corr_S1_spec_loop_Tr1 = df_tr_1_spec_loop["resp_total_corr"].sum()
        Corr_S1_sub_loop_Tr1 = df_tr_1_sub_loop["resp_total_corr"].sum()
        Corr_S1_rule_loop_Tr1 = df_tr_1_rule_loop["resp_total_corr"].sum()
        Corr_S1_gen_loop_Tr1 = df_tr_1_gen_loop["resp_total_corr"].sum()
        
        loop_n_1[iterations] = df_tr_1_spec_loop["resp_total_corr"].sum() #loop with its response value
        loop_n_2[iterations] = df_tr_1_sub_loop["resp_total_corr"].sum()
        loop_n_3[iterations] = df_tr_1_rule_loop["resp_total_corr"].sum()
        loop_n_4[iterations] = df_tr_1_gen_loop["resp_total_corr"].sum()
    

    df_Corr_S1_spec_per_loop = pd.DataFrame.from_dict(loop_n_1,  orient='index')
    df_Corr_S1_spec_per_loop = df_Corr_S1_spec_per_loop.transpose()
    df_Corr_S1_spec_per_loop.columns = ["Corr_S1_spec_1", "Corr_S1_spec_2", "Corr_S1_spec_3", "Corr_S1_spec_4", "Corr_S1_spec_5", "Corr_S1_spec_6"]
    
    df_Corr_S1_sub_per_loop = pd.DataFrame.from_dict(loop_n_2,  orient='index')
    df_Corr_S1_sub_per_loop = df_Corr_S1_sub_per_loop.transpose()
    df_Corr_S1_sub_per_loop.columns = ["Corr_S1_sub_1", "Corr_S1_sub_2", "Corr_S1_sub_3", "Corr_S1_sub_4", "Corr_S1_sub_5", "Corr_S1_sub_6"]
    
    df_Corr_S1_rule_per_loop = pd.DataFrame.from_dict(loop_n_3,  orient='index')
    df_Corr_S1_rule_per_loop = df_Corr_S1_rule_per_loop.transpose()
    df_Corr_S1_rule_per_loop.columns = ["Corr_S1_rule_1", "Corr_S1_rule_2", "Corr_S1_rule_3", "Corr_S1_rule_4", "Corr_S1_rule_5", "Corr_S1_rule_6"]
    
    df_Corr_S1_gen_per_loop = pd.DataFrame.from_dict(loop_n_4,  orient='index')
    df_Corr_S1_gen_per_loop = df_Corr_S1_gen_per_loop.transpose()
    df_Corr_S1_gen_per_loop.columns = ["Corr_S1_gen_1", "Corr_S1_gen_2", "Corr_S1_gen_3", "Corr_S1_gen_4", "Corr_S1_gen_5", "Corr_S1_gen_6"]
    
    loop_n_1.clear() 
    loop_n_2.clear()
    loop_n_3.clear()
    loop_n_4.clear()
    
    #Response Time RT TR1
    #clean the dataset and remove filler and incorrect trials
    df_training_1 = df_training_1.drop(df_training_1[df_training_1['trial_type'] == "filler"].index)
    df_training_1 = df_training_1.loc[(df_training_1['resp_total_corr'] == 1)]
    
    df_tr_1_spec_tot = df_training_1.loc[(df_training_1['conditions'] == "spec")]
    RT_Spec_Tot_Tr1 = np.mean(df_tr_1_spec_tot["resp_total_time"])
    
    df_tr_1_sub_tot = df_training_1.loc[(df_training_1['conditions'] == "subrule")]
    RT_Sub_Tot_Tr1 = np.mean(df_tr_1_sub_tot["resp_total_time"])

    df_tr_1_rule_tot = df_training_1.loc[(df_training_1['conditions'] == "rule")]
    RT_Rule_Tot_Tr1 = np.mean(df_tr_1_rule_tot["resp_total_time"])

    df_tr_1_gen_tot = df_training_1.loc[(df_training_1['conditions'] == "general")]
    RT_Gen_Tot_Tr1 = np.mean(df_tr_1_gen_tot["resp_total_time"])
    
    loop_n_5 = {}
    
    # RT for each loop TR1
    for iterations in range(0,6):
        df_training_1_loop = df_training_1.loc[df_training_1['repeat_training_loop1.thisRepN'] == iterations]
        
        df_tr_1_spec_loop = df_training_1_loop.loc[(df_training_1_loop['conditions'] == "spec")]
        df_tr_1_sub_loop = df_training_1_loop.loc[(df_training_1_loop['conditions'] == "subrule")]
        df_tr_1_rule_loop = df_training_1_loop.loc[(df_training_1_loop['conditions'] == "rule")]
        df_tr_1_gen_loop = df_training_1_loop.loc[(df_training_1_loop['conditions'] == "general")]
        
        RT_Spec_loop_Tr1 = np.mean(df_tr_1_spec_loop["resp_total_time"])
        RT_Sub_loop_Tr1 = np.mean(df_tr_1_sub_loop["resp_total_time"])
        RT_Rule_loop_Tr1 = np.mean(df_tr_1_rule_loop["resp_total_time"])
        RT_Gen_loop_Tr1 = np.mean(df_tr_1_gen_loop["resp_total_time"])
        
        loop_n_1[iterations] = np.mean(df_tr_1_spec_loop["resp_total_time"]) #loop with its response value
        loop_n_2[iterations] = np.mean(df_tr_1_sub_loop["resp_total_time"])
        loop_n_3[iterations] = np.mean(df_tr_1_rule_loop["resp_total_time"]) #loop with its response value
        loop_n_4[iterations] = np.mean(df_tr_1_gen_loop["resp_total_time"])
        
        loop_n_5[iterations] = np.mean(df_training_1_loop["resp_total_time"])
    
    df_RT1_spec_per_loop = pd.DataFrame.from_dict(loop_n_1,  orient='index')
    df_RT1_spec_per_loop = df_RT1_spec_per_loop.transpose()
    df_RT1_spec_per_loop.columns = ["RT1_spec_1", "RT1_spec_2", "RT1_spec_3", "RT1_spec_4", "RT1_spec_5", "RT1_spec_6"]
    
    df_RT1_sub_per_loop = pd.DataFrame.from_dict(loop_n_2,  orient='index')
    df_RT1_sub_per_loop = df_RT1_sub_per_loop.transpose()
    df_RT1_sub_per_loop.columns = ["RT1_sub_1", "RT1_sub_2", "RT1_sub_3", "RT1_sub_4", "RT1_sub_5", "RT1_sub_6"]
    
    df_RT1_rule_per_loop = pd.DataFrame.from_dict(loop_n_3,  orient='index')
    df_RT1_rule_per_loop = df_RT1_rule_per_loop.transpose()
    df_RT1_rule_per_loop.columns = ["RT1_rule_1", "RT1_rule_2", "RT1_rule_3", "RT1_rule_4", "RT1_rule_5", "RT1_rule_6"]
    
    df_RT1_gen_per_loop = pd.DataFrame.from_dict(loop_n_4,  orient='index')
    df_RT1_gen_per_loop = df_RT1_gen_per_loop.transpose()
    df_RT1_gen_per_loop.columns = ["RT1_gen_1", "RT1_gen_2", "RT1_gen_3", "RT1_gen_4", "RT1_gen_5", "RT1_gen_6"]
  
    df_RT1_per_loop = pd.DataFrame.from_dict(loop_n_5,  orient='index')
    df_RT1_per_loop = df_RT1_per_loop.transpose()
    df_RT1_per_loop.columns = ["RT1_1_Tot", "RT1_2_Tot", "RT1_3_Tot", "RT1_4_Tot", "RT1_5_Tot", "RT1_6_Tot"]
     
    loop_n_1.clear() #empty the previous dictionary
    loop_n_2.clear()
    loop_n_3.clear()
    loop_n_4.clear()
    loop_n_5.clear()
    
    #Onset Time OT (first press) TR1
    df_ot_1_spec_tot = df_training_1.loc[(df_training_1['conditions'] == "spec")]
    OT_Spec_Tot_Tr1 = np.mean(df_ot_1_spec_tot["resp1.rt"])
    
    df_ot_1_sub_tot = df_training_1.loc[(df_training_1['conditions'] == "subrule")]
    OT_Sub_Tot_Tr1 = np.mean(df_ot_1_sub_tot["resp1.rt"])

    df_ot_1_rule_tot = df_training_1.loc[(df_training_1['conditions'] == "rule")]
    OT_Rule_Tot_Tr1 = np.mean(df_ot_1_rule_tot["resp1.rt"])

    df_ot_1_gen_tot = df_training_1.loc[(df_training_1['conditions'] == "general")]
    OT_Gen_Tot_Tr1 = np.mean(df_ot_1_gen_tot["resp1.rt"])
    
    # OT for each loop TR1
    for iterations in range(0,6):
        df_training_1_loop = df_training_1.loc[df_tr['repeat_training_loop1.thisRepN'] == iterations]
        
        df_ot_1_spec_loop = df_training_1_loop.loc[(df_training_1_loop['conditions'] == "spec")]
        df_ot_1_sub_loop = df_training_1_loop.loc[(df_training_1_loop['conditions'] == "subrule")]
        df_ot_1_rule_loop = df_training_1_loop.loc[(df_training_1_loop['conditions'] == "rule")]
        df_ot_1_gen_loop = df_training_1_loop.loc[(df_training_1_loop['conditions'] == "general")]
        
        OT_Spec_loop_Tr1 = np.mean(df_ot_1_spec_loop["resp1.rt"])
        OT_Sub_loop_Tr1 = np.mean(df_ot_1_sub_loop["resp1.rt"])
        OT_Rule_loop_Tr1 = np.mean(df_ot_1_rule_loop["resp1.rt"])
        OT_Gen_loop_Tr1 = np.mean(df_ot_1_gen_loop["resp1.rt"])
        
        loop_n_1[iterations] = np.mean(df_ot_1_spec_loop["resp1.rt"]) #loop with its response value
        loop_n_2[iterations] = np.mean(df_ot_1_sub_loop["resp1.rt"])
        loop_n_3[iterations] = np.mean(df_ot_1_rule_loop["resp1.rt"])
        loop_n_4[iterations] = np.mean(df_ot_1_gen_loop["resp1.rt"])
        loop_n_5[iterations] = np.mean(df_training_1_loop["resp1.rt"])

    df_OT1_spec_per_loop = pd.DataFrame.from_dict(loop_n_1,  orient='index')
    df_OT1_spec_per_loop = df_OT1_spec_per_loop.transpose()
    df_OT1_spec_per_loop.columns = ["OT1_spec_1", "OT1_spec_2", "OT1_spec_3", "OT1_spec_4", "OT1_spec_5", "OT1_spec_6"]
    
    df_OT1_sub_per_loop = pd.DataFrame.from_dict(loop_n_2,  orient='index')
    df_OT1_sub_per_loop = df_OT1_sub_per_loop.transpose()
    df_OT1_sub_per_loop.columns = ["OT1_sub_1", "OT1_sub_2", "OT1_sub_3", "OT1_sub_4", "OT1_sub_5", "OT1_sub_6"]
    
    df_OT1_rule_per_loop = pd.DataFrame.from_dict(loop_n_3,  orient='index')
    df_OT1_rule_per_loop = df_OT1_rule_per_loop.transpose()
    df_OT1_rule_per_loop.columns = ["OT1_rule_1", "OT1_rule_2", "OT1_rule_3", "OT1_rule_4", "OT1_rule_5", "OT1_rule_6"]
    
    df_OT1_gen_per_loop = pd.DataFrame.from_dict(loop_n_4,  orient='index')
    df_OT1_gen_per_loop = df_OT1_gen_per_loop.transpose()
    df_OT1_gen_per_loop.columns = ["OT1_gen_1", "OT1_gen_2", "OT1_gen_3", "OT1_gen_4", "OT1_gen_5", "OT1_gen_6"]
    
    df_OT1_per_loop = pd.DataFrame.from_dict(loop_n_5,  orient='index')
    df_OT1_per_loop = df_OT1_per_loop.transpose()
    df_OT1_per_loop.columns = ["OT1_1_Tot", "OT1_2_Tot", "OT1_3_Tot", "OT1_4_Tot", "OT1_5_Tot", "OT1_6_Tot"]

    loop_n_1.clear() #empty the previous dictionary
    loop_n_2.clear()
    loop_n_3.clear() 
    loop_n_4.clear()
    loop_n_5.clear()
    
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
    
    for iterations in range(0,6):
        df_training_2_loop = df_training_2.loc[df_training_2['repeat_training_loop1b.thisRepN'] == iterations]
        df_training_2_loop_wo = df_training_2_wo.loc[df_training_2_wo['repeat_training_loop1b.thisRepN'] == iterations]
        
        Total_resp_Seq_loop_te = df_training_2_loop["resp_total_corr"].sum()
        
        loop_n_5[iterations] = df_training_2_loop_wo["resp_total_corr"].sum()        
        
        loop_n_0[iterations] = df_training_2_loop["resp_total_corr"].sum()
       
        df_tr_2_spec_loop = df_training_2_loop.loc[(df_training_2_loop['conditions'] == "spec")]
        df_tr_2_sub_loop = df_training_2_loop.loc[(df_training_2_loop['conditions'] == "subrule")]
        df_tr_2_rule_loop = df_training_2_loop.loc[(df_training_2_loop['conditions'] == "rule")]
        df_tr_2_gen_loop = df_training_2_loop.loc[(df_training_2_loop['conditions'] == "general")]
          
        loop_n_1[iterations] = df_tr_2_spec_loop["resp_total_corr"].sum() #loop with its response value
        loop_n_2[iterations] = df_tr_2_sub_loop["resp_total_corr"].sum()
        loop_n_3[iterations] = df_tr_2_rule_loop["resp_total_corr"].sum()
        loop_n_4[iterations] = df_tr_2_gen_loop["resp_total_corr"].sum()
                 
    df_Corr_Seq2_per_loop = pd.DataFrame.from_dict(loop_n_0,  orient='index')
    df_Corr_Seq2_per_loop= df_Corr_Seq2_per_loop.transpose()
    df_Corr_Seq2_per_loop.columns = ["Corr_S2_1", "Corr_S2_2", "Corr_S2_3", "Corr_S2_4", "Corr_S2_5", "Corr_S2_6"]    
   
    df_Corr_S2_spec_per_loop = pd.DataFrame.from_dict(loop_n_1,  orient='index')
    df_Corr_S2_spec_per_loop = df_Corr_S2_spec_per_loop.transpose()
    df_Corr_S2_spec_per_loop.columns = ["Corr_S2_spec_1", "Corr_S2_spec_2", "Corr_S2_spec_3", "Corr_S2_spec_4", "Corr_S2_spec_5", "Corr_S2_spec_6"]
    
    df_Corr_S2_sub_per_loop = pd.DataFrame.from_dict(loop_n_2,  orient='index')
    df_Corr_S2_sub_per_loop = df_Corr_S2_sub_per_loop.transpose()
    df_Corr_S2_sub_per_loop.columns = ["Corr_S2_sub_1", "Corr_S2_sub_2", "Corr_S2_sub_3", "Corr_S2_sub_4", "Corr_S2_sub_5", "Corr_S2_sub_6"]
    
    df_Corr_S2_rule_per_loop = pd.DataFrame.from_dict(loop_n_3,  orient='index')
    df_Corr_S2_rule_per_loop = df_Corr_S2_rule_per_loop.transpose()
    df_Corr_S2_rule_per_loop.columns = ["Corr_S2_rule_1", "Corr_S2_rule_2", "Corr_S2_rule_3", "Corr_S2_rule_4", "Corr_S2_rule_5", "Corr_S2_rule_6"]
    
    df_Corr_S2_gen_per_loop = pd.DataFrame.from_dict(loop_n_4,  orient='index')
    df_Corr_S2_gen_per_loop = df_Corr_S2_gen_per_loop.transpose()
    df_Corr_S2_gen_per_loop.columns = ["Corr_S2_gen_1", "Corr_S2_gen_2", "Corr_S2_gen_3", "Corr_S2_gen_4", "Corr_S2_gen_5", "Corr_S2_gen_6"]   
    
    df_Corr_Seq2_per_loop_wo = pd.DataFrame.from_dict(loop_n_5,  orient='index')
    df_Corr_Seq2_per_loop_wo = df_Corr_Seq2_per_loop_wo.transpose()   
    df_Corr_Seq2_per_loop_wo.columns = ["Corr_S2_1_wo", "Corr_S2_2_wo", "Corr_S2_3_wo", "Corr_S2_4_wo", "Corr_S2_5_wo", "Corr_S2_6_wo"]

    loop_n_1.clear() #empty the previous dictionary
    loop_n_2.clear()
    loop_n_3.clear() #empty the previous dictionary
    loop_n_4.clear()    
    loop_n_5.clear()    
    
    #Response Time RT TR2
    #clean the dataset and remove filler and incorrect trials
    
    df_tr_2_spec_tot = df_training_2.loc[(df_training_2['conditions'] == "spec")]
    RT_Spec_Tot_Tr2 = np.mean(df_tr_2_spec_tot["resp_total_time"])
    
    df_tr_2_sub_tot = df_training_2.loc[(df_training_2['conditions'] == "subrule")]
    RT_Sub_Tot_Tr2 = np.mean(df_tr_2_sub_tot["resp_total_time"])

    df_tr_2_rule_tot = df_training_2.loc[(df_training_2['conditions'] == "rule")]
    RT_Rule_Tot_Tr2 = np.mean(df_tr_2_rule_tot["resp_total_time"])

    df_tr_2_gen_tot = df_training_2.loc[(df_training_2['conditions'] == "general")]
    RT_Gen_Tot_Tr2 = np.mean(df_tr_2_gen_tot["resp_total_time"])

    # RT for each loop TR2
    for iterations in range(0,6):
        df_training_2_loop = df_training_2_wo.loc[df_tr['repeat_training_loop1b.thisRepN'] == iterations]
 
        df_tr_2_spec_loop = df_training_2_loop.loc[(df_training_2_loop['conditions'] == "spec")]
        df_tr_2_sub_loop = df_training_2_loop.loc[(df_training_2_loop['conditions'] == "subrule")]
        df_tr_2_rule_loop = df_training_2_loop.loc[(df_training_2_loop['conditions'] == "rule")]
        df_tr_2_gen_loop = df_training_2_loop.loc[(df_training_2_loop['conditions'] == "general")]
        
        RT_Spec_loop_Tr2 = np.mean(df_tr_2_spec_loop["resp_total_time"])
        RT_Sub_loop_Tr2 = np.mean(df_tr_2_sub_loop["resp_total_time"])
        RT_Rule_loop_Tr2 = np.mean(df_tr_2_rule_loop["resp_total_time"])
        RT_Gen_loop_Tr2 = np.mean(df_tr_2_gen_loop["resp_total_time"])
        
        loop_n_1[iterations] = np.mean(df_tr_2_spec_loop["resp_total_time"]) #loop with its response value
        loop_n_2[iterations] = np.mean(df_tr_2_sub_loop["resp_total_time"])
        loop_n_3[iterations] = np.mean(df_tr_2_rule_loop["resp_total_time"]) #loop with its response value
        loop_n_4[iterations] = np.mean(df_tr_2_gen_loop["resp_total_time"])
        loop_n_5[iterations] = np.mean(df_training_2_loop["resp_total_time"])
    
    df_RT2_spec_per_loop = pd.DataFrame.from_dict(loop_n_1,  orient='index')
    df_RT2_spec_per_loop = df_RT2_spec_per_loop.transpose()
    df_RT2_spec_per_loop.columns = ["RT2_spec_1", "RT2_spec_2", "RT2_spec_3", "RT2_spec_4", "RT2_spec_5", "RT2_spec_6"]
 
    df_RT2_sub_per_loop = pd.DataFrame.from_dict(loop_n_2,  orient='index')
    df_RT2_sub_per_loop = df_RT2_sub_per_loop.transpose()
    df_RT2_sub_per_loop.columns = ["RT2_sub_1", "RT2_sub_2", "RT2_sub_3", "RT2_sub_4", "RT2_sub_5", "RT2_sub_6"]
    
    df_RT2_rule_per_loop = pd.DataFrame.from_dict(loop_n_3,  orient='index')
    df_RT2_rule_per_loop = df_RT2_rule_per_loop.transpose()
    df_RT2_rule_per_loop.columns = ["RT2_rule_1", "RT2_rule_2", "RT2_rule_3", "RT2_rule_4", "RT2_rule_5", "RT2_rule_6"]
    
    df_RT2_gen_per_loop = pd.DataFrame.from_dict(loop_n_4,  orient='index')
    df_RT2_gen_per_loop = df_RT2_gen_per_loop.transpose()
    df_RT2_gen_per_loop.columns = ["RT2_gen_1", "RT2_gen_2", "RT2_gen_3", "RT2_gen_4", "RT2_gen_5", "RT2_gen_6"]
        
    df_RT2_per_loop = pd.DataFrame.from_dict(loop_n_5,  orient='index')
    df_RT2_per_loop = df_RT2_per_loop.transpose()
    df_RT2_per_loop.columns = ["RT2_1_Tot", "RT2_2_Tot", "RT2_3_Tot", "RT2_4_Tot", "RT2_5_Tot", "RT2_6_Tot"]    
    
    loop_n_1.clear()
    loop_n_2.clear()
    loop_n_3.clear() 
    loop_n_4.clear()
    loop_n_5.clear()

    #Onset Time OT (first press) TR2
    df_ot_2_spec_tot = df_training_2.loc[(df_training_2['conditions'] == "spec")]
    OT_Spec_Tot_Tr2 = np.mean(df_ot_2_spec_tot["resp1b.rt"])
    
    df_ot_2_sub_tot = df_training_2.loc[(df_training_2['conditions'] == "subrule")]
    OT_Sub_Tot_Tr2 = np.mean(df_ot_2_sub_tot["resp1b.rt"])

    df_ot_2_rule_tot = df_training_2.loc[(df_training_2['conditions'] == "rule")]
    OT_Rule_Tot_Tr2 = np.mean(df_ot_2_rule_tot["resp1b.rt"])

    df_ot_2_gen_tot = df_training_2.loc[(df_training_2['conditions'] == "general")]
    OT_Gen_Tot_Tr2 = np.mean(df_ot_2_gen_tot["resp1b.rt"])
    
    # OT for each loop TR2
    for iterations in range(0,6):
        df_training_2_loop = df_training_2_wo.loc[df_training_2['repeat_training_loop1b.thisRepN'] == iterations]
        
        df_ot_2_spec_loop = df_training_2_loop.loc[(df_training_2_loop['conditions'] == "spec")]
        df_ot_2_sub_loop = df_training_2_loop.loc[(df_training_2_loop['conditions'] == "subrule")]
        df_ot_2_rule_loop = df_training_2_loop.loc[(df_training_2_loop['conditions'] == "rule")]
        df_ot_2_gen_loop = df_training_2_loop.loc[(df_training_2_loop['conditions'] == "general")]
        
        OT_Spec_loop_Tr2 = np.mean(df_ot_2_spec_loop["resp1b.rt"])
        OT_Sub_loop_Tr2 = np.mean(df_ot_2_sub_loop["resp1b.rt"])
        OT_Rule_loop_Tr2 = np.mean(df_ot_2_rule_loop["resp1b.rt"])
        OT_Gen_loop_Tr2 = np.mean(df_ot_2_gen_loop["resp1b.rt"])
        
        loop_n_1[iterations] = np.mean(df_ot_2_spec_loop["resp1b.rt"]) #loop with its response value
        loop_n_2[iterations] = np.mean(df_ot_2_sub_loop["resp1b.rt"])
        loop_n_3[iterations] = np.mean(df_ot_2_rule_loop["resp1b.rt"])
        loop_n_4[iterations] = np.mean(df_ot_2_gen_loop["resp1b.rt"])
        loop_n_5[iterations] = np.mean(df_training_2_loop["resp1b.rt"])

    
    df_OT2_spec_per_loop = pd.DataFrame.from_dict(loop_n_1,  orient='index')
    df_OT2_spec_per_loop = df_OT2_spec_per_loop.transpose()
    df_OT2_spec_per_loop.columns = ["OT2_spec_1", "OT2_spec_2", "OT2_spec_3", "OT2_spec_4", "OT2_spec_5", "OT2_spec_6"]
    
    df_OT2_sub_per_loop = pd.DataFrame.from_dict(loop_n_2,  orient='index')
    df_OT2_sub_per_loop = df_OT2_sub_per_loop.transpose()
    df_OT2_sub_per_loop.columns = ["OT2_sub_1", "OT2_sub_2", "OT2_sub_3", "OT2_sub_4", "OT2_sub_5", "OT2_sub_6"]
    
    df_OT2_rule_per_loop = pd.DataFrame.from_dict(loop_n_3,  orient='index')
    df_OT2_rule_per_loop = df_OT2_rule_per_loop.transpose()
    df_OT2_rule_per_loop.columns = ["OT2_rule_1", "OT2_rule_2", "OT2_rule_3", "OT2_rule_4", "OT2_rule_5", "OT2_rule_6"]
    
    df_OT2_gen_per_loop = pd.DataFrame.from_dict(loop_n_4,  orient='index')
    df_OT2_gen_per_loop = df_OT2_gen_per_loop.transpose()
    df_OT2_gen_per_loop.columns = ["OT2_gen_1", "OT2_gen_2", "OT2_gen_3", "OT2_gen_4", "OT2_gen_5", "OT2_gen_6"]
    
    df_OT2_per_loop = pd.DataFrame.from_dict(loop_n_5,  orient='index')
    df_OT2_per_loop = df_OT2_per_loop.transpose()
    df_OT2_per_loop.columns = ["OT2_1_Tot", "OT2_2_Tot", "OT2_3_Tot", "OT2_4_Tot", "OT2_5_Tot", "OT2_6_Tot"]

    loop_n_0.clear()      
    loop_n_1.clear() 
    loop_n_2.clear()
    loop_n_3.clear() 
    loop_n_4.clear()
    loop_n_5.clear()
    
    #concatenate all the dataframes created so far
    result = pd.concat([df_Corr_R_spec_per_loop, df_Corr_R_sub_per_loop, df_Corr_R_rule_per_loop, 
                        df_Corr_R_gen_per_loop, df_Corr_S1_spec_per_loop, df_Corr_S1_sub_per_loop, 
                        df_Corr_S1_rule_per_loop ,df_Corr_S1_gen_per_loop, 
                        df_RT1_spec_per_loop, df_RT1_sub_per_loop, df_RT1_rule_per_loop,
                        df_RT1_gen_per_loop, df_OT1_spec_per_loop, df_OT1_sub_per_loop, 
                        df_OT1_rule_per_loop,df_OT1_gen_per_loop, df_RT2_spec_per_loop, 
                        df_RT2_sub_per_loop, df_RT2_rule_per_loop,df_RT2_gen_per_loop, 
                        df_OT2_spec_per_loop, df_OT2_sub_per_loop, df_OT2_rule_per_loop,
                        df_OT2_gen_per_loop, df_Corr_S2_spec_per_loop, df_Corr_S2_sub_per_loop, 
                        df_Corr_S2_rule_per_loop, df_Corr_S2_gen_per_loop, df_Corr_R_per_loop, 
                        df_Corr_R_per_loop_wo, df_Corr_Seq1_per_loop, df_Corr_Seq1_per_loop_wo,
                        df_Corr_Seq2_per_loop, df_Corr_Seq2_per_loop_wo, df_OT1_per_loop, df_RT1_per_loop,
                        df_RT2_per_loop, df_OT2_per_loop], axis=1, sort=False)
   
    # add columns contaning the following values
    result["Subj_tr"] = participant_number_tr

    result["Corr_R_Tot"] = Corr_R_Tot
    result["Corr_S1_Tot"] = Corr_S_Tot
    
    result["Corr_R_Tot_wo"] = Corr_R_Tot_wo
    result["Corr_S1_Tot_wo"] = Corr_S_Tot_wo
    
    result["Corr_R_spec"] = df_Corr_R_spec_per_loop.sum(axis=1)
    result["Corr_R_sub"] = df_Corr_R_sub_per_loop.sum(axis=1)
    result["Corr_R_rule"] = df_Corr_R_rule_per_loop.sum(axis=1)
    result["Corr_R_gen"] = df_Corr_R_gen_per_loop.sum(axis=1)

    result["Corr_S1_spec"] = df_Corr_S1_spec_per_loop.sum(axis=1)
    result["Corr_S1_sub"] = df_Corr_S1_sub_per_loop.sum(axis=1)
    result["Corr_S1_rule"] = df_Corr_S1_rule_per_loop.sum(axis=1)
    result["Corr_S1_gen"] = df_Corr_S1_gen_per_loop.sum(axis=1)    
    
    result["Corr_S2_Tot"] = Total_resp_Seq_2
    result["Corr_S2_Tot_wo"] = Corr_S2_Tot_wo
    
    result["RT1_spec_Tot"] = RT_Spec_Tot_Tr1
    result["RT1_sub_Tot"] = RT_Sub_Tot_Tr1
    result["RT1_rule_Tot"] = RT_Rule_Tot_Tr1
    result["RT1_gen_Tot"] = RT_Gen_Tot_Tr1
    
    result["OT1_spec_Tot"] = OT_Spec_Tot_Tr1
    result["OT1_sub_Tot"] = OT_Sub_Tot_Tr1
    result["OT1_rule_Tot"] = OT_Rule_Tot_Tr1
    result["OT1_gen_Tot"] = OT_Gen_Tot_Tr1        


    result["RT2_spec_Tot"] = RT_Spec_Tot_Tr2
    result["RT2_sub_Tot"] = RT_Sub_Tot_Tr2
    result["RT2_rule_Tot"] = RT_Rule_Tot_Tr2
    result["RT2_gen_Tot"] = RT_Gen_Tot_Tr2
    
    result["OT2_spec_Tot"] = OT_Spec_Tot_Tr2
    result["OT2_sub_Tot"] = OT_Sub_Tot_Tr2
    result["OT2_rule_Tot"] = OT_Rule_Tot_Tr2
    result["OT2_gen_Tot"] = OT_Gen_Tot_Tr2     
    
    result["Corr_S2_spec"] = df_Corr_S2_spec_per_loop.sum(axis=1)
    result["Corr_S2_sub"] = df_Corr_S2_sub_per_loop.sum(axis=1)
    result["Corr_S2_rule"] = df_Corr_S2_rule_per_loop.sum(axis=1)
    result["Corr_S2_gen"] = df_Corr_S2_gen_per_loop.sum(axis=1)    
    
    # append this to the empty dataframe created at the beginning
    df_result_tr = df_result_tr.append(result)


###############################
############ TEST ############
##############################

# read the files in the test folder    
pattern = re.compile(r'(\d*)_Mental')
participant_numbers =  ["; ".join(pattern.findall(item)) for item in UPDATED_files_te]

for UPDATED_file_te in UPDATED_files_te: #UPDATED_file_te = each participant (number)
    
    participant_number_te = [int(s) for s in re.findall(r'(\d*)_Mental', UPDATED_file_te)]
    str_number_te = ' '.join(map(str, participant_number_te))
    
    # with fillers
    df_te = pd.read_csv(data_folder_test + UPDATED_file_te,  header=0)
    
    df_te = df_te.loc[df_te['file_n'] >= 1]
    file_values = df_te.file_n.unique() 
    df_te["file_n"].replace({file_values[0]: int("1"), file_values[1]: int("2"),file_values[2]: int("3"),
         file_values[3]: int("4"), file_values[4]: int("5"), file_values[5]: int("6")}, inplace=True)
   
    df_te = df_te.loc[(df_te['trial_type'] == "experimental")|(df_te['trial_type'] == "filler")]  
    Corr_S3_Tot = df_te["resp_total_corr"].sum()      
    
    # without filler
    df_te_wo = df_te.loc[(df_te['file_n'] >= 1) & (df_te['trial_type'] == "experimental")]
    Corr_S3_Tot_wo = df_te_wo["resp_total_corr"].sum() 
    
    # range changes because in the excel the column "file_n" has a value going NOT from 0 to 6 as in training 
    # BUT from 1 to 7
    for iterations in range(1,7):
        df_te_loop = df_te.loc[df_te['file_n'] == iterations]
        df_te_loop_wo = df_te_wo.loc[df_te_wo['file_n'] == iterations]
        
        Total_resp_Seq_loop_te = df_te_loop["resp_total_corr"].sum()

        loop_n_0[iterations] = df_te_loop["resp_total_corr"].sum()
        loop_n_5[iterations] = df_te_loop_wo["resp_total_corr"].sum()

        df_te_spec_loop = df_te_loop.loc[(df_te_loop['conditions'] == "spec")]
        df_te_sub_loop = df_te_loop.loc[(df_te_loop['conditions'] == "subrule")]
        df_te_rule_loop = df_te_loop.loc[(df_te_loop['conditions'] == "rule")]
        df_te_gen_loop = df_te_loop.loc[(df_te_loop['conditions'] == "general")]
          
        loop_n_1[iterations] = df_te_spec_loop["resp_total_corr"].sum() #loop with its response value
        loop_n_2[iterations] = df_te_sub_loop["resp_total_corr"].sum()
        loop_n_3[iterations] = df_te_rule_loop["resp_total_corr"].sum()
        loop_n_4[iterations] = df_te_gen_loop["resp_total_corr"].sum()
    
    # S1 = TR1 / S2 = TR2 / S3 = Test
    # the same goes for RT and OT
    df_Corr_S3_per_loop = pd.DataFrame.from_dict(loop_n_0,  orient='index')
    df_Corr_S3_per_loop = df_Corr_S3_per_loop.transpose()
    df_Corr_S3_per_loop.columns = ["Corr_S3_1", "Corr_S3_2", "Corr_S3_3", "Corr_S3_4", "Corr_S3_5", "Corr_S3_6"]    
   
    df_Corr_S3_spec_per_loop = pd.DataFrame.from_dict(loop_n_1,  orient='index')
    df_Corr_S3_spec_per_loop = df_Corr_S3_spec_per_loop.transpose()
    df_Corr_S3_spec_per_loop.columns = ["Corr_S3_spec_1", "Corr_S3_spec_2", "Corr_S3_spec_3", "Corr_S3_spec_4", "Corr_S3_spec_5", "Corr_S3_spec_6"]
    
    df_Corr_S3_sub_per_loop = pd.DataFrame.from_dict(loop_n_2,  orient='index')
    df_Corr_S3_sub_per_loop = df_Corr_S3_sub_per_loop.transpose()
    df_Corr_S3_sub_per_loop.columns = ["Corr_S3_sub_1", "Corr_S3_sub_2", "Corr_S3_sub_3", "Corr_S3_sub_4", "Corr_S3_sub_5", "Corr_S3_sub_6"]
    
    df_Corr_S3_rule_per_loop= pd.DataFrame.from_dict(loop_n_3,  orient='index')
    df_Corr_S3_rule_per_loop = df_Corr_S3_rule_per_loop.transpose()
    df_Corr_S3_rule_per_loop.columns = ["Corr_S3_rule_1", "Corr_S3_rule_2", "Corr_S3_rule_3", "Corr_S3_rule_4", "Corr_S3_rule_5", "Corr_S3_rule_6"]
    
    df_Corr_S3_gen_per_loop = pd.DataFrame.from_dict(loop_n_4,  orient='index')
    df_Corr_S3_gen_per_loop = df_Corr_S3_gen_per_loop.transpose()
    df_Corr_S3_gen_per_loop.columns = ["Corr_S3_gen_1", "Corr_S3_gen_2", "Corr_S3_gen_3", "Corr_S3_gen_4", "Corr_S3_gen_5", "Corr_S3_gen_6"]   
             
    df_Corr_S3_per_loop_wo = pd.DataFrame.from_dict(loop_n_5,  orient='index')
    df_Corr_S3_per_loop_wo = df_Corr_S3_per_loop_wo.transpose()
    df_Corr_S3_per_loop_wo.columns = ["Corr_S3_1_wo", "Corr_S3_2_wo", "Corr_S3_3_wo", "Corr_S3_4_wo", "Corr_S3_5_wo", "Corr_S3_6_wo"]

    loop_n_0.clear()
    loop_n_1.clear() #empty the previous dictionary
    loop_n_2.clear()
    loop_n_3.clear() 
    loop_n_4.clear()    
    loop_n_5.clear()
    
    #Response Time RT TEST
    #clean the dataset and remove filler and incorrect trials
    df_te = df_te.loc[(df_te['resp_total_corr'] == 1)]
    df_te = df_te.loc[(df_te['trial_type'] == "experimental")]

    df_te_spec_tot = df_te.loc[(df_te['conditions'] == "spec")]
    RT_Spec_Tot_Te = np.mean(df_te_spec_tot["resp_total_time"])
    
    df_te_sub_tot = df_te.loc[(df_te['conditions'] == "subrule")]
    RT_Sub_Tot_Te = np.mean(df_te_sub_tot["resp_total_time"])

    df_te_rule_tot = df_te.loc[(df_te['conditions'] == "rule")]
    RT_Rule_Tot_Te = np.mean(df_te_rule_tot["resp_total_time"])

    df_te_gen_tot = df_te.loc[(df_te['conditions'] == "general")]
    RT_Gen_Tot_Te = np.mean(df_te_gen_tot["resp_total_time"])

    # RT for each loop TE
    for iterations in range(1,7):
        df_te_loop = df_te.loc[df_te['file_n'] == iterations]
        
        df_te_spec_loop = df_te_loop.loc[(df_te_loop['conditions'] == "spec")]
        df_te_sub_loop = df_te_loop.loc[(df_te_loop['conditions'] == "subrule")]
        df_te_rule_loop = df_te_loop.loc[(df_te_loop['conditions'] == "rule")]
        df_te_gen_loop = df_te_loop.loc[(df_te_loop['conditions'] == "general")]
       
        RT_Spec_loop_Te = np.mean(df_te_spec_loop["resp_total_time"])
        RT_Sub_loop_Te = np.mean(df_te_sub_loop["resp_total_time"])
        RT_Rule_loop_Te = np.mean(df_te_rule_loop["resp_total_time"])
        RT_Gen_loop_Te = np.mean(df_te_gen_loop["resp_total_time"])
        
        loop_n_1[iterations] = np.mean(df_te_spec_loop["resp_total_time"]) #loop with its response value
        loop_n_2[iterations] = np.mean(df_te_sub_loop["resp_total_time"])
        loop_n_3[iterations] = np.mean(df_te_rule_loop["resp_total_time"])
        loop_n_4[iterations] = np.mean(df_te_gen_loop["resp_total_time"])
        loop_n_5[iterations] = np.mean(df_te_loop["resp_total_time"])
    
    df_RT3_spec_per_loop = pd.DataFrame.from_dict(loop_n_1,  orient='index')
    df_RT3_spec_per_loop = df_RT3_spec_per_loop.transpose()
    df_RT3_spec_per_loop.columns = ["RT3_spec_1", "RT3_spec_2", "RT3_spec_3", "RT3_spec_4", "RT3_spec_5", "RT3_spec_6"]
 
    df_RT3_sub_per_loop = pd.DataFrame.from_dict(loop_n_2,  orient='index')
    df_RT3_sub_per_loop = df_RT3_sub_per_loop.transpose()
    df_RT3_sub_per_loop.columns = ["RT3_sub_1", "RT3_sub_2", "RT3_sub_3", "RT3_sub_4", "RT3_sub_5", "RT3_sub_6"]
    
    df_RT3_rule_per_loop = pd.DataFrame.from_dict(loop_n_3,  orient='index')
    df_RT3_rule_per_loop = df_RT3_rule_per_loop.transpose()
    df_RT3_rule_per_loop.columns = ["RT3_rule_1", "RT3_rule_2", "RT3_rule_3", "RT3_rule_4", "RT3_rule_5", "RT3_rule_6"]
    
    df_RT3_gen_per_loop = pd.DataFrame.from_dict(loop_n_4,  orient='index')
    df_RT3_gen_per_loop = df_RT3_gen_per_loop.transpose()
    df_RT3_gen_per_loop.columns = ["RT3_gen_1", "RT3_gen_2", "RT3_gen_3", "RT3_gen_4", "RT3_gen_5", "RT3_gen_6"]
    
    df_RT3_per_loop = pd.DataFrame.from_dict(loop_n_5,  orient='index')
    df_RT3_per_loop = df_RT3_per_loop.transpose()
    df_RT3_per_loop.columns = ["RT3_1_Tot", "RT3_2_Tot", "RT3_3_Tot", "RT3_4_Tot", "RT3_5_Tot", "RT3_6_Tot"]
     
    loop_n_1.clear()
    loop_n_2.clear()
    loop_n_3.clear() 
    loop_n_4.clear()
    loop_n_5.clear()
    
    #Onset Time OT (first press) TE
    df_ot_te_spec_tot = df_te.loc[(df_te['conditions'] == "spec")]
    OT_Spec_Tot_Te = np.mean(df_ot_te_spec_tot["resp1.rt"])
    
    df_ot_te_sub_tot = df_te.loc[(df_te['conditions'] == "subrule")]
    OT_Sub_Tot_Te = np.mean(df_ot_te_sub_tot["resp1.rt"])

    df_ot_te_rule_tot = df_te.loc[(df_te['conditions'] == "rule")]
    OT_Rule_Tot_Te = np.mean(df_ot_te_rule_tot["resp1.rt"])

    df_ot_te_gen_tot = df_te.loc[(df_te['conditions'] == "general")]
    OT_Gen_Tot_Te = np.mean(df_ot_te_gen_tot["resp1.rt"])
        
    # OT for each loop TE
    for iterations in range(1,7):
        df_te_loop = df_te.loc[df_te['file_n'] == iterations]
        
        df_ot_te_spec_loop = df_te_loop.loc[(df_te_loop['conditions'] == "spec")]
        df_ot_te_sub_loop = df_te_loop.loc[(df_te_loop['conditions'] == "subrule")]
        df_ot_te_rule_loop = df_te_loop.loc[(df_te_loop['conditions'] == "rule")]
        df_ot_te_gen_loop = df_te_loop.loc[(df_te_loop['conditions'] == "general")]
        
        OT_Spec_loop_Te = np.mean(df_ot_te_spec_loop["resp1.rt"])
        OT_Sub_loop_Te = np.mean(df_ot_te_sub_loop["resp1.rt"])
        OT_Rule_loop_Te = np.mean(df_ot_te_rule_loop["resp1.rt"])
        OT_Gen_loop_Te = np.mean(df_ot_te_gen_loop["resp1.rt"])
        
        loop_n_1[iterations] = np.mean(df_ot_te_spec_loop["resp1.rt"]) #loop with its response value
        loop_n_2[iterations] = np.mean(df_ot_te_sub_loop["resp1.rt"])
        loop_n_3[iterations] = np.mean(df_ot_te_rule_loop["resp1.rt"])
        loop_n_4[iterations] = np.mean(df_ot_te_gen_loop["resp1.rt"])
        loop_n_5[iterations] = np.mean(df_te_loop["resp1.rt"])

    df_OT3_spec_per_loop = pd.DataFrame.from_dict(loop_n_1,  orient='index')
    df_OT3_spec_per_loop = df_OT3_spec_per_loop.transpose()
    df_OT3_spec_per_loop.columns = ["OT3_spec_1", "OT3_spec_2", "OT3_spec_3", "OT3_spec_4", "OT3_spec_5", "OT3_spec_6"]
    
    df_OT3_sub_per_loop = pd.DataFrame.from_dict(loop_n_2,  orient='index')
    df_OT3_sub_per_loop = df_OT3_sub_per_loop.transpose()
    df_OT3_sub_per_loop.columns = ["OT3_sub_1", "OT3_sub_2", "OT3_sub_3", "OT3_sub_4", "OT3_sub_5", "OT3_sub_6"]
    
    df_OT3_rule_per_loop = pd.DataFrame.from_dict(loop_n_3,  orient='index')
    df_OT3_rule_per_loop = df_OT3_rule_per_loop.transpose()
    df_OT3_rule_per_loop.columns = ["OT3_rule_1", "OT3_rule_2", "OT3_rule_3", "OT3_rule_4", "OT3_rule_5", "OT3_rule_6"]
    
    df_OT3_gen_per_loop = pd.DataFrame.from_dict(loop_n_4,  orient='index')
    df_OT3_gen_per_loop = df_OT3_gen_per_loop.transpose()
    df_OT3_gen_per_loop.columns = ["OT3_gen_1", "OT3_gen_2", "OT3_gen_3", "OT3_gen_4", "OT3_gen_5", "OT3_gen_6"]
    
    df_OT3_per_loop = pd.DataFrame.from_dict(loop_n_5,  orient='index')
    df_OT3_per_loop = df_OT3_per_loop.transpose()
    df_OT3_per_loop.columns = ["OT3_1_Tot", "OT3_2_Tot", "OT3_3_Tot", "OT3_4_Tot", "OT3_5_Tot", "OT3_6_Tot"]

    loop_n_1.clear() 
    loop_n_2.clear()
    loop_n_3.clear() 
    loop_n_4.clear()    
    loop_n_5.clear()
    
    #concatenate all the dataframes created for test
    result_te = pd.concat([df_Corr_S3_per_loop, df_Corr_S3_spec_per_loop, df_Corr_S3_sub_per_loop, df_Corr_S3_rule_per_loop, df_Corr_S3_gen_per_loop,
                          df_RT3_spec_per_loop, df_RT3_sub_per_loop, df_RT3_rule_per_loop, df_RT3_gen_per_loop,
                          df_OT3_spec_per_loop, df_OT3_sub_per_loop, df_OT3_rule_per_loop, df_OT3_gen_per_loop, 
                          df_Corr_S3_per_loop_wo, df_RT3_per_loop, df_RT3_spec_per_loop, df_OT3_per_loop], axis=1, sort=False)
    
    #add the following columns and values
    result_te["Corr_S3_Tot"] = Corr_S3_Tot
    result_te["Corr_S3_Tot_wo"] = Corr_S3_Tot_wo
    
    result_te["Corr_S3_spec"] = df_Corr_S3_spec_per_loop.sum(axis=1)
    result_te["Corr_S3_sub"] = df_Corr_S3_sub_per_loop.sum(axis=1)
    result_te["Corr_S3_rule"] = df_Corr_S3_rule_per_loop.sum(axis=1)
    result_te["Corr_S3_gen"] = df_Corr_S3_gen_per_loop.sum(axis=1)    
        
    result_te["RT3_spec_Tot"] = RT_Spec_Tot_Te
    result_te["RT3_sub_Tot"] = RT_Sub_Tot_Te
    result_te["RT3_rule_Tot"] = RT_Rule_Tot_Te
    result_te["RT3_gen_Tot"] = RT_Gen_Tot_Te
    
    result_te["OT3_spec_Tot"] = OT_Spec_Tot_Te
    result_te["OT3_sub_Tot"] = OT_Sub_Tot_Te
    result_te["OT3_rule_Tot"] = OT_Rule_Tot_Te
    result_te["OT3_gen_Tot"] = OT_Gen_Tot_Te
    
    result_te["Subj_te"] = str_number_te
    result_te["Group"] = group
    
    df_result_te = df_result_te.append(result_te)


####################################
############ END SCRIPT ############
###################################   

#training + test  = concate the two results dataframes  
result_total = pd.concat([df_result_tr, df_result_te], axis=1, sort=False)

#reorder all the columns in the preferred order
result_total = result_total[["Subj_tr" , "Subj_te" , "Group" , "Corr_R_Tot" , "Corr_R_Tot_wo" , "Corr_R_1" , 
                             "Corr_R_1_wo" , "Corr_R_2" , "Corr_R_2_wo" , "Corr_R_3" , "Corr_R_3_wo" , "Corr_R_4" , 
                             "Corr_R_4_wo" , "Corr_R_5" , "Corr_R_5_wo" , "Corr_R_6" , "Corr_R_6_wo" , 
                             "Corr_R_spec_1" , "Corr_R_spec_2" , "Corr_R_spec_3" , "Corr_R_spec_4" , "Corr_R_spec_5" ,
                             "Corr_R_spec_6" , "Corr_R_spec" , "Corr_R_sub_1" , "Corr_R_sub_2" , "Corr_R_sub_3" , 
                             "Corr_R_sub_4" , "Corr_R_sub_5" , "Corr_R_sub_6" , "Corr_R_sub" , "Corr_R_rule_1" , 
                             "Corr_R_rule_2" , "Corr_R_rule_3" , "Corr_R_rule_4" , "Corr_R_rule_5" , 
                             "Corr_R_rule_6" , "Corr_R_rule" , "Corr_R_gen_1" , "Corr_R_gen_2" , "Corr_R_gen_3" , 
                             "Corr_R_gen_4" , "Corr_R_gen_5" , "Corr_R_gen_6" , "Corr_R_gen" , "Corr_S1_Tot" , 
                             "Corr_S1_Tot_wo" , "Corr_S1_1" , "Corr_S1_1_wo" , "Corr_S1_2" , "Corr_S1_2_wo" , 
                             "Corr_S1_3" , "Corr_S1_3_wo" , "Corr_S1_4" , "Corr_S1_4_wo" , "Corr_S1_5" , 
                             "Corr_S1_5_wo" , "Corr_S1_6" , "Corr_S1_6_wo" , "Corr_S1_spec_1" , "Corr_S1_spec_2" , 
                             "Corr_S1_spec_3" , "Corr_S1_spec_4" , "Corr_S1_spec_5" , "Corr_S1_spec_6" , 
                             "Corr_S1_spec" , "Corr_S1_sub_1" , "Corr_S1_sub_2" , "Corr_S1_sub_3" , "Corr_S1_sub_4" ,
                             "Corr_S1_sub_5" , "Corr_S1_sub_6" , "Corr_S1_sub" , "Corr_S1_rule_1" , "Corr_S1_rule_2" ,
                             "Corr_S1_rule_3" , "Corr_S1_rule_4" , "Corr_S1_rule_5" , "Corr_S1_rule_6" , "Corr_S1_rule" ,
                             "Corr_S1_gen_1" , "Corr_S1_gen_2" , "Corr_S1_gen_3" , "Corr_S1_gen_4" , "Corr_S1_gen_5" , 
                             "Corr_S1_gen_6" , "Corr_S1_gen" , "RT1_1_Tot" , "RT1_2_Tot" , "RT1_3_Tot" , "RT1_4_Tot" , 
                             "RT1_5_Tot" , "RT1_6_Tot" , "RT1_spec_1" , "RT1_spec_2" , "RT1_spec_3" , "RT1_spec_4" , 
                             "RT1_spec_5" , "RT1_spec_6" , "RT1_spec_Tot" , "RT1_sub_1" , "RT1_sub_2" , "RT1_sub_3" , 
                             "RT1_sub_4" , "RT1_sub_5" , "RT1_sub_6" , "RT1_sub_Tot" , "RT1_rule_1" , "RT1_rule_2" , 
                             "RT1_rule_3" , "RT1_rule_4" , "RT1_rule_5" , "RT1_rule_6" , "RT1_rule_Tot" , "RT1_gen_1" ,
                             "RT1_gen_2" , "RT1_gen_3" , "RT1_gen_4" , "RT1_gen_5" , "RT1_gen_6" , "RT1_gen_Tot" , 
                             "OT1_1_Tot" , "OT1_2_Tot" , "OT1_3_Tot" , "OT1_4_Tot" , "OT1_5_Tot" , "OT1_6_Tot" , 
                             "OT1_spec_1" , "OT1_spec_2" , "OT1_spec_3" , "OT1_spec_4" , "OT1_spec_5" , "OT1_spec_6" , 
                             "OT1_spec_Tot" , "OT1_sub_1" , "OT1_sub_2" , "OT1_sub_3" , "OT1_sub_4" , "OT1_sub_5" , 
                             "OT1_sub_6" , "OT1_sub_Tot" , "OT1_rule_1" , "OT1_rule_2" , "OT1_rule_3" , "OT1_rule_4" , 
                             "OT1_rule_5" , "OT1_rule_6" , "OT1_rule_Tot" , "OT1_gen_1" , "OT1_gen_2" , "OT1_gen_3" , 
                             "OT1_gen_4" , "OT1_gen_5" , "OT1_gen_6" , "OT1_gen_Tot" , "Corr_S2_Tot" , "Corr_S2_Tot_wo" , 
                             "Corr_S2_1" , "Corr_S2_1_wo" , "Corr_S2_2" , "Corr_S2_2_wo" , "Corr_S2_3" , "Corr_S2_3_wo" , 
                             "Corr_S2_4" , "Corr_S2_4_wo" , "Corr_S2_5" , "Corr_S2_5_wo" , "Corr_S2_6" , "Corr_S2_6_wo" , 
                             "Corr_S2_spec_1" , "Corr_S2_spec_2" , "Corr_S2_spec_3" , "Corr_S2_spec_4" , "Corr_S2_spec_5" , 
                             "Corr_S2_spec_6" , "Corr_S2_spec" , "Corr_S2_sub_1" , "Corr_S2_sub_2" , "Corr_S2_sub_3" , 
                             "Corr_S2_sub_4" , "Corr_S2_sub_5" , "Corr_S2_sub_6" , "Corr_S2_sub" , "Corr_S2_rule_1" ,
                             "Corr_S2_rule_2" , "Corr_S2_rule_3" , "Corr_S2_rule_4" , "Corr_S2_rule_5" , "Corr_S2_rule_6" , 
                             "Corr_S2_rule" , "Corr_S2_gen_1" , "Corr_S2_gen_2" , "Corr_S2_gen_3" , "Corr_S2_gen_4" , 
                             "Corr_S2_gen_5" , "Corr_S2_gen_6" , "Corr_S2_gen" , "RT2_1_Tot" , "RT2_2_Tot" , "RT2_3_Tot" ,
                             "RT2_4_Tot" , "RT2_5_Tot" , "RT2_6_Tot" , "RT2_spec_1" , "RT2_spec_2" , "RT2_spec_3" , 
                             "RT2_spec_4" , "RT2_spec_5" , "RT2_spec_6" , "RT2_spec_Tot" , "RT2_sub_1" , "RT2_sub_2" , 
                             "RT2_sub_3" , "RT2_sub_4" , "RT2_sub_5" , "RT2_sub_6" , "RT2_sub_Tot" , "RT2_rule_1" , 
                             "RT2_rule_2" , "RT2_rule_3" , "RT2_rule_4" , "RT2_rule_5" , "RT2_rule_6" , "RT2_rule_Tot" , 
                             "RT2_gen_1" , "RT2_gen_2" , "RT2_gen_3" , "RT2_gen_4" , "RT2_gen_5" , "RT2_gen_6" , 
                             "RT2_gen_Tot" , "OT2_1_Tot" , "OT2_2_Tot" , "OT2_3_Tot" , "OT2_4_Tot" , "OT2_5_Tot" , 
                             "OT2_6_Tot" , "OT2_spec_1" , "OT2_spec_2" , "OT2_spec_3" , "OT2_spec_4" , "OT2_spec_5" , 
                             "OT2_spec_6" , "OT2_spec_Tot" , "OT2_sub_1" , "OT2_sub_2" , "OT2_sub_3" , "OT2_sub_4" , 
                             "OT2_sub_5" , "OT2_sub_6" , "OT2_sub_Tot" , "OT2_rule_1" , "OT2_rule_2" , "OT2_rule_3" , 
                             "OT2_rule_4" , "OT2_rule_5" , "OT2_rule_6" , "OT2_rule_Tot" , "OT2_gen_1" , "OT2_gen_2" , 
                             "OT2_gen_3" , "OT2_gen_4" , "OT2_gen_5" , "OT2_gen_6" , "OT2_gen_Tot" , "Corr_S3_Tot" , 
                             "Corr_S3_Tot_wo" , "Corr_S3_1" , "Corr_S3_1_wo" , "Corr_S3_2" , "Corr_S3_2_wo" , "Corr_S3_3" , 
                             "Corr_S3_3_wo" , "Corr_S3_4" , "Corr_S3_4_wo" , "Corr_S3_5" , "Corr_S3_5_wo" , "Corr_S3_6" , 
                             "Corr_S3_6_wo" , "Corr_S3_spec_1" , "Corr_S3_spec_2" , "Corr_S3_spec_3" , "Corr_S3_spec_4" , 
                             "Corr_S3_spec_5" , "Corr_S3_spec_6" , "Corr_S3_spec" , "Corr_S3_sub_1" , "Corr_S3_sub_2" , 
                             "Corr_S3_sub_3" , "Corr_S3_sub_4" , "Corr_S3_sub_5" , "Corr_S3_sub_6" , "Corr_S3_sub" , 
                             "Corr_S3_rule_1" , "Corr_S3_rule_2" , "Corr_S3_rule_3" , "Corr_S3_rule_4" , "Corr_S3_rule_5" , 
                             "Corr_S3_rule_6" , "Corr_S3_rule" , "Corr_S3_gen_1" , "Corr_S3_gen_2" , "Corr_S3_gen_3" , 
                             "Corr_S3_gen_4" , "Corr_S3_gen_5" , "Corr_S3_gen_6" , "Corr_S3_gen" , "RT3_1_Tot" , 
                             "RT3_2_Tot" , "RT3_3_Tot" , "RT3_4_Tot" , "RT3_5_Tot" , "RT3_6_Tot" , "RT3_spec_1" , 
                             "RT3_spec_2" , "RT3_spec_3" , "RT3_spec_4" , "RT3_spec_5" , "RT3_spec_6" , "RT3_spec_Tot" , 
                             "RT3_sub_1" , "RT3_sub_2" , "RT3_sub_3" , "RT3_sub_4" , "RT3_sub_5" , "RT3_sub_6" , 
                             "RT3_sub_Tot" , "RT3_rule_1" , "RT3_rule_2" , "RT3_rule_3" , "RT3_rule_4" , "RT3_rule_5" , 
                             "RT3_rule_6" , "RT3_rule_Tot" , "RT3_gen_1" , "RT3_gen_2" , "RT3_gen_3" , "RT3_gen_4" , 
                             "RT3_gen_5" , "RT3_gen_6" , "RT3_gen_Tot" , "OT3_1_Tot" , "OT3_2_Tot" , "OT3_3_Tot" , 
                             "OT3_4_Tot" , "OT3_5_Tot" , "OT3_6_Tot" , "OT3_spec_1" , "OT3_spec_2" , "OT3_spec_3" , 
                             "OT3_spec_4" , "OT3_spec_5" , "OT3_spec_6" , "OT3_spec_Tot" , "OT3_sub_1" , "OT3_sub_2" , 
                             "OT3_sub_3" , "OT3_sub_4" , "OT3_sub_5" , "OT3_sub_6" , "OT3_sub_Tot" , "OT3_rule_1" , 
                             "OT3_rule_2" , "OT3_rule_3" , "OT3_rule_4" , "OT3_rule_5" , "OT3_rule_6" , "OT3_rule_Tot" , 
                             "OT3_gen_1" , "OT3_gen_2" , "OT3_gen_3" , "OT3_gen_4" , "OT3_gen_5" , "OT3_gen_6" , 
                             "OT3_gen_Tot"]]

# print name of columns / just to check that everything is there
#columns_names = list(result_total.columns.values.tolist())
#with open('columns_names.txt', 'w') as f:
#    for columns_name in columns_names:
#        print >> f, columns_name

result_total.to_csv("/data/pt_02312/results/results.csv", index = False, header=True)

############################
############################
#####ANALYSES ON TEST#######
############################
############################

result_total = pd.read_csv("/data/pt_02312/results/results.csv",  header=0)

result_total_spec = result_total[['Subj_te','RT3_spec_Tot','OT3_spec_Tot']]
result_total_sub = result_total[['Subj_te','RT3_sub_Tot', 'OT3_sub_Tot']]
result_total_rule = result_total[['Subj_te','RT3_rule_Tot', 'OT3_rule_Tot']]
result_total_gen = result_total[['Subj_te','RT3_gen_Tot', 'OT3_gen_Tot']]

#arrange the dataframe from a horizonatal to a vertical position
for row in result_total:  
    result_total_spec['condition'] = "spec" 
    result_total_sub['condition'] = "sub" 
    result_total_rule['condition'] = "rule" 
    result_total_gen['condition'] = "gen" 

#rename columns
#append all dataframes together
dfs=[result_total_spec, result_total_sub, result_total_rule, result_total_gen]

rddf = pd.DataFrame()
for df_x in dfs:
    df_x.columns = ["Id", "RT", "OT", "condition"]
    rddf = rddf.append(df_x)

rddf_OT = rddf[['Id','condition', "OT"]]
rddf_RT = rddf[['Id','condition', "RT"]]

#### identifying outliers ####
## boxplots
for rt_ot in ["RT","OT"]:
    
    if rt_ot == "RT":
        df_rt_ot = rddf_RT
    elif rt_ot == "OT":
        df_rt_ot = rddf_OT
        
    pl_cond = sns.boxplot(x = "condition", y = rt_ot, data = df_rt_ot) #condition
    plt.figure()
    #pl_subj = sns.boxplot(x = "Id", y = rt_ot, data = df_rt_ot) #subject

    #this loop creates a plot for each condition x id
    #for cond in df_rt_ot["condition"].unique():  
     #   rddf_sub = df_rt_ot[df_rt_ot.condition == cond]
     #   plt.figure() #this creates a new figure on which your plot will appear
     #   pl_subj_cond = sns.boxplot(x = "Id", y = rt_ot, data = rddf_sub) # subject x condition
     #   pl_subj_cond.set_title(cond)
     #   plt.figure()
        
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
    for subj in rddf["Id"].unique():
        outliers_focus = "subject"
        rddf_sub_out = rddf[rddf.Id == subj]
        df_without_out_subj, df_only_out_subj = iqr_func(rddf_sub_out)

        df_3.append(df_without_out_subj)
        df_4.append(df_only_out_subj)

    #SUBJECTS X CONDITION
    for subj in rddf["Id"].unique(): 
        for cond in rddf["condition"].unique():
            outliers_focus = "subject and condition"
            rddf_sub_out = rddf[rddf.condition == cond]
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
    #pl_subj_after = sns.boxplot(x = "Id", y = rt_ot, data = df_rt_ot_2) #subject

    
    #this loop creates a plot for each condition x id
    #for cond in df_rt_ot_3["condition"].unique():  
    #    rddf_sub_after = df_rt_ot_3[df_rt_ot_3.condition == cond]
    #    plt.figure() #this creates a new figure on which your plot will appear
    #    pl_subj_cond_after = sns.boxplot(x = "Id", y = rt_ot, data = rddf_sub_after) # subject x condition
    #    pl_subj_cond_after.set_title(cond)
    
#########################    
#### Analysis for RT ####
#########################    

# based on https://link.springer.com/article/10.3758/s13428-011-0172-y/tables/2

#get the number of subjects
subjects = rddf['Id'].nunique()

#ONE-WAY REPEATED MEASURES ANOVA
#Print results with relevant SSs MS df F p

for diffdf in ["condRT", "scRT"]:
    
    if diffdf == "condRT":
        df = df_without_out_cond_conc_RT
    elif diffdf == "scRT":
        df = df_without_out_subj_cond_conc_RT

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
        elif row['condition'] == "sub":
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
    elif diffdf == "scOT":
        df = df_without_out_subj_cond_conc_OT

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
        elif row['condition'] == "sub":
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
    ###perform post-hocs ###
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

