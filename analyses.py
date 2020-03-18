import pandas as pd
import os
import re
import numpy as np 

data_folder_training = "/home/raid2/papitto/Desktop/PsychoPy/MRep_2020_backup/MRep_training_backup/data/"
data_folder_test = "/home/raid2/papitto/Desktop/PsychoPy/MRep_2020_backup/MRep_test_backup/data/"

df_result_tr = pd.DataFrame([])

#read all the files in a folders and creat a list
filenames_tr = os.listdir(data_folder_training)

#create a list only with files UPDATED
UPDATED_files_tr = []

for filename_tr in filenames_tr:
    if filename_tr.endswith("UPDATED.csv"):
        UPDATED_files_tr.append(filename_tr)

#extract the number of the participants
pattern = re.compile(r'(\d*)_Mental')
participant_numbers =  ["; ".join(pattern.findall(item)) for item in UPDATED_files_tr]

#read all the UPDATED files one at the time
for UPDATED_file_tr in UPDATED_files_tr:
    participant_number = [int(s) for s in re.findall(r'(\d*)_Mental', UPDATED_file_tr)]
    str_number = ' '.join(map(str, participant_number)) 
    
    df_tr = pd.read_csv(UPDATED_file_tr,  header=0)
    
    df_exp_fil_trials = df_tr.loc[(df_tr['trial_type'] == "experimental")|(df_tr['trial_type'] == "filler")]
    
    
    ###############################
    ######### TRAINING 1 #########
    ##############################

    # ALL LOOPS TOGETHER
    df_training_1 = df_exp_fil_trials.loc[df_tr['repeat_training_loop1.thisRepN'] >= 0]
    df_training_1.dropna(how='all', axis=1)
    Corr_R_Tot = df_training_1["resp_R.corr"].sum() #number of correct relationship answers (incl. fillers) - all loops
    Corr_S_Tot = df_training_1["resp_total_corr"].sum() #number of correct sequences (incl. fillers) - all loops
    
    loop_n_1 = {} #create empty ditctionary
    loop_n_2 = {}
    loop_n_3 = {} #create empty ditctionary
    loop_n_4 = {}
    
    for iterations_1 in range(0,6):
        df_training_1_loop = df_exp_fil_trials.loc[df_tr['repeat_training_loop1.thisRepN'] == iterations_1]
        df_training_1_loop.dropna(how='all', axis=1)
        Total_resp_R_loop = df_training_1_loop["resp_R.corr"].sum() #number of correct relationship answers (incl. fillers) - all loops
        Total_resp_Seq_loop = df_training_1_loop["resp_total_corr"].sum()
        loop_n_1[iterations_1] = df_training_1_loop["resp_R.corr"].sum() #loop with its response value
        loop_n_2[iterations_1] = df_training_1_loop["resp_total_corr"].sum()
        
    df_corr_R_per_loop = pd.DataFrame.from_dict(loop_n_1,  orient='index')
    df_corr_R_per_loop = df_corr_R_per_loop.transpose()
    df_corr_R_per_loop.columns = ["corr_R_1", "corr_R_2", "corr_R_3", "corr_R_4", "corr_R_5", "corr_R_6"]
     
    df_corr_Seq_per_loop = pd.DataFrame.from_dict(loop_n_2,  orient='index')
    df_corr_Seq_per_loop = df_corr_Seq_per_loop.transpose()
    df_corr_R_per_loop.columns = ["corr_Sa_1", "corr_Sa_2", "corr_Sa_3", "corr_Sa_4", "corr_Sa_5", "corr_Sa_6"]    
   
    loop_n_1.clear() #empty the previous dictionary
    loop_n_2.clear()
    
    # Corr_R for each loop TR1
    for iterations_0 in range(0,6):
        df_training_1_loop = df_training_1.loc[df_tr['repeat_training_loop1.thisRepN'] == iterations_0]
        df_tr_1_spec_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "spec")]
        df_tr_1_sub_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "subrule")]
        df_tr_1_rule_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "rule")]
        df_tr_1_gen_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "general")]
        
        Corr_R_Spec_loop_Tr1 = df_tr_1_spec_loop["resp_R.corr"].sum()
        Corr_R_Sub_loop_Tr1 = df_tr_1_sub_loop["resp_R.corr"].sum()
        Corr_R_Rule_loop_Tr1 = df_tr_1_rule_loop["resp_R.corr"].sum()
        Corr_R_Gen_loop_Tr1 = df_tr_1_gen_loop["resp_R.corr"].sum()
        
        loop_n_1[iterations_0] = df_tr_1_spec_loop["resp_R.corr"].sum() #loop with its response value
        loop_n_2[iterations_0] = df_tr_1_sub_loop["resp_R.corr"].sum()
        loop_n_3[iterations_0] = df_tr_1_rule_loop["resp_R.corr"].sum()
        loop_n_4[iterations_0] = df_tr_1_gen_loop["resp_R.corr"].sum()
        
        Corr_S_Spec_loop_Tr1 = df_tr_1_spec_loop["resp_total_corr"].sum()
        Corr_S_Sub_loop_Tr1 = df_tr_1_sub_loop["resp_total_corr"].sum()
        Corr_S_Rule_loop_Tr1 = df_tr_1_rule_loop["resp_total_corr"].sum()
        Corr_S_Gen_loop_Tr1 = df_tr_1_gen_loop["resp_total_corr"].sum()
        
        loop_n_1[iterations_0] = df_tr_1_spec_loop["resp_total_corr"].sum() #loop with its response value
        loop_n_2[iterations_0] = df_tr_1_sub_loop["resp_total_corr"].sum()
        loop_n_3[iterations_0] = df_tr_1_rule_loop["resp_total_corr"].sum()
        loop_n_4[iterations_0] = df_tr_1_gen_loop["resp_total_corr"].sum()
    
    df_corr_R_spec_per_loop = pd.DataFrame.from_dict(loop_n_1,  orient='index')
    df_corr_R_spec_per_loop = df_corr_R_spec_per_loop.transpose()
    df_corr_R_spec_per_loop.columns = ["Corr_R_spec1_1", "Corr_R_spec1_2", "Corr_R_spec1_3", "Corr_R_spec1_4", "Corr_R_spec1_5", "Corr_R_spec1_6"]
    
    df_corr_R_sub_per_loop = pd.DataFrame.from_dict(loop_n_2,  orient='index')
    df_corr_R_sub_per_loop = df_corr_R_sub_per_loop.transpose()
    df_corr_R_sub_per_loop.columns = ["Corr_R_sub1_1", "Corr_R_sub1_2", "Corr_R_sub1_3", "Corr_R_sub1_4", "Corr_R_sub1_5", "Corr_R_sub1_6"]
    
    df_corr_R_rule_per_loop = pd.DataFrame.from_dict(loop_n_3,  orient='index')
    df_corr_R_rule_per_loop = df_corr_R_rule_per_loop.transpose()
    df_corr_R_rule_per_loop.columns = ["Corr_R_rule1_1", "Corr_R_rule1_2", "Corr_R_rule1_3", "Corr_R_rule1_4", "Corr_R_rule1_5", "Corr_R_rule1_6"]
    
    df_corr_R_gen_per_loop = pd.DataFrame.from_dict(loop_n_4,  orient='index')
    df_corr_R_gen_per_loop = df_corr_R_gen_per_loop.transpose()
    df_corr_R_gen_per_loop.columns = ["Corr_R_gen1_1", "Corr_R_gen1_2", "Corr_R_gen1_3", "Corr_R_gen1_4", "Corr_R_gen1_5", "Corr_R_gen1_6"]
    
    df_corr_S_spec_per_loop = pd.DataFrame.from_dict(loop_n_1,  orient='index')
    df_corr_S_spec_per_loop = df_corr_S_spec_per_loop.transpose()
    df_corr_S_spec_per_loop.columns = ["Corr_S_spec1_1", "Corr_S_spec1_2", "Corr_S_spec1_3", "Corr_S_spec1_4", "Corr_S_spec1_5", "Corr_S_spec1_6"]
    
    df_corr_S_sub_per_loop = pd.DataFrame.from_dict(loop_n_2,  orient='index')
    df_corr_S_sub_per_loop = df_corr_S_sub_per_loop.transpose()
    df_corr_S_sub_per_loop.columns = ["Corr_S_sub1_1", "Corr_S_sub1_2", "Corr_S_sub1_3", "Corr_S_sub1_4", "Corr_S_sub1_5", "Corr_S_sub1_6"]
    
    df_corr_S_rule_per_loop = pd.DataFrame.from_dict(loop_n_3,  orient='index')
    df_corr_S_rule_per_loop = df_corr_S_rule_per_loop.transpose()
    df_corr_S_rule_per_loop.columns = ["Corr_S_rule1_1", "Corr_S_rule1_2", "Corr_S_rule1_3", "Corr_S_rule1_4", "Corr_S_rule1_5", "Corr_S_rule1_6"]
    
    df_corr_S_gen_per_loop = pd.DataFrame.from_dict(loop_n_4,  orient='index')
    df_corr_S_gen_per_loop = df_corr_S_gen_per_loop.transpose()
    df_corr_S_gen_per_loop.columns = ["Corr_S_gen1_1", "Corr_S_gen1_2", "Corr_S_gen1_3", "Corr_S_gen1_4", "Corr_S_gen1_5", "Corr_S_gen1_6"]
    
    loop_n_1.clear() #empty the previous dictionary
    loop_n_2.clear()
    loop_n_3.clear() #empty the previous dictionary
    loop_n_4.clear()
    
    
    #Response Time RT TR1
    #clean the dataset and remove filler and incorrect trials
    df_training_1 = df_training_1.drop(df_training_1[df_training_1['trial_type'] == "filler"].index)
    df_training_1 = df_training_1.drop(df_training_1[df_training_1['resp_total_corr'] == 0].index)
    
    df_tr_1_spec_tot = df_training_1.loc[(df_training_1['conditions'] == "spec")]
    RT_Spec_Tot_Tr1 = np.mean(df_tr_1_spec_tot["resp_total_time"])
    
    df_tr_1_sub_tot = df_training_1.loc[(df_training_1['conditions'] == "subrule")]
    RT_Sub_Tot_Tr1 = np.mean(df_tr_1_sub_tot["resp_total_time"])

    df_tr_1_rule_tot = df_training_1.loc[(df_training_1['conditions'] == "rule")]
    RT_Rule_Tot_Tr1 = np.mean(df_tr_1_rule_tot["resp_total_time"])

    df_tr_1_gen_tot = df_training_1.loc[(df_training_1['conditions'] == "general")]
    RT_Gen_Tot_Tr1 = np.mean(df_tr_1_gen_tot["resp_total_time"])
    
    # RT for each loop TR1
    for iterations_3 in range(0,6):
        df_training_1_loop = df_training_1.loc[df_tr['repeat_training_loop1.thisRepN'] == iterations_3]
        df_tr_1_spec_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "spec")]
        df_tr_1_sub_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "subrule")]
        df_tr_1_rule_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "rule")]
        df_tr_1_gen_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "general")]
        
        RT_Spec_loop_Tr1 = np.mean(df_tr_1_spec_loop["resp_total_time"])
        RT_Sub_loop_Tr1 = np.mean(df_tr_1_sub_loop["resp_total_time"])
        RT_Rule_loop_Tr1 = np.mean(df_tr_1_rule_loop["resp_total_time"])
        RT_Gen_loop_Tr1 = np.mean(df_tr_1_gen_loop["resp_total_time"])
        
        loop_n_1[iterations_3] = np.mean(df_tr_1_spec_loop["resp_total_time"]) #loop with its response value
        loop_n_2[iterations_3] = np.mean(df_tr_1_sub_loop["resp_total_time"])
        loop_n_3[iterations_3] = np.mean(df_tr_1_rule_loop["resp_total_time"]) #loop with its response value
        loop_n_4[iterations_3] = np.mean(df_tr_1_gen_loop["resp_total_time"])
    
    df_RT_spec_per_loop = pd.DataFrame.from_dict(loop_n_1,  orient='index')
    df_RT_spec_per_loop = df_RT_spec_per_loop.transpose()
    df_RT_spec_per_loop.columns = ["RT_spec1_1", "RT_spec1_2", "RT_spec1_3", "RT_spec1_4", "RT_spec1_5", "RT_spec1_6"]
    
    df_RT_sub_per_loop = pd.DataFrame.from_dict(loop_n_2,  orient='index')
    df_RT_sub_per_loop = df_RT_sub_per_loop.transpose()
    df_RT_sub_per_loop.columns = ["RT_sub1_1", "RT_sub1_2", "RT_sub1_3", "RT_sub1_4", "RT_sub1_5", "RT_sub1_6"]
    
    df_RT_rule_per_loop = pd.DataFrame.from_dict(loop_n_3,  orient='index')
    df_RT_rule_per_loop = df_RT_rule_per_loop.transpose()
    df_RT_rule_per_loop.columns = ["RT_rule1_1", "RT_rule1_2", "RT_rule1_3", "RT_rule1_4", "RT_rule1_5", "RT_rule1_6"]
    
    df_RT_gen_per_loop = pd.DataFrame.from_dict(loop_n_4,  orient='index')
    df_RT_gen_per_loop = df_RT_gen_per_loop.transpose()
    df_RT_gen_per_loop.columns = ["RT_gen1_1", "RT_gen1_2", "RT_gen1_3", "RT_gen1_4", "RT_gen1_5", "RT_gen1_6"]
     
    loop_n_1.clear() #empty the previous dictionary
    loop_n_2.clear()
    loop_n_3.clear() #empty the previous dictionary
    loop_n_4.clear()
    
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
    for iterations_4 in range(0,6):
        df_training_1_loop = df_training_1.loc[df_tr['repeat_training_loop1.thisRepN'] == iterations_4]
        df_ot_1_spec_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "spec")]
        df_ot_1_sub_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "subrule")]
        df_ot_1_rule_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "rule")]
        df_ot_1_gen_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "general")]
        
        OT_Spec_loop_Tr1 = np.mean(df_ot_1_spec_loop["resp1.rt"])
        OT_Sub_loop_Tr1 = np.mean(df_ot_1_sub_loop["resp1.rt"])
        OT_Rule_loop_Tr1 = np.mean(df_ot_1_rule_loop["resp1.rt"])
        OT_Gen_loop_Tr1 = np.mean(df_ot_1_gen_loop["resp1.rt"])
        
        loop_n_1[iterations_4] = np.mean(df_ot_1_spec_loop["resp1.rt"]) #loop with its response value
        loop_n_2[iterations_4] = np.mean(df_ot_1_sub_loop["resp1.rt"])
        loop_n_3[iterations_4] = np.mean(df_ot_1_rule_loop["resp1.rt"])
        loop_n_4[iterations_4] = np.mean(df_ot_1_gen_loop["resp1.rt"])
    
    df_OT_spec_per_loop = pd.DataFrame.from_dict(loop_n_1,  orient='index')
    df_OT_spec_per_loop = df_OT_spec_per_loop.transpose()
    df_OT_spec_per_loop.columns = ["OT_spec1_1", "OT_spec1_2", "OT_spec1_3", "OT_spec1_4", "OT_spec1_5", "OT_spec1_6"]
    
    df_OT_sub_per_loop = pd.DataFrame.from_dict(loop_n_2,  orient='index')
    df_OT_sub_per_loop = df_OT_sub_per_loop.transpose()
    df_OT_sub_per_loop.columns = ["OT_sub1_1", "OT_sub1_2", "OT_sub1_3", "OT_sub1_4", "OT_sub1_5", "OT_sub1_6"]
    
    df_OT_rule_per_loop = pd.DataFrame.from_dict(loop_n_3,  orient='index')
    df_OT_rule_per_loop = df_OT_rule_per_loop.transpose()
    df_OT_rule_per_loop.columns = ["OT_rule1_1", "OT_rule1_2", "OT_rule1_3", "OT_rule1_4", "OT_rule1_5", "OT_rule1_6"]
    
    df_OT_gen_per_loop = pd.DataFrame.from_dict(loop_n_4,  orient='index')
    df_OT_gen_per_loop = df_OT_gen_per_loop.transpose()
    df_OT_gen_per_loop.columns = ["OT_gen1_1", "OT_gen1_2", "OT_gen1_3", "OT_gen1_4", "OT_gen1_5", "OT_gen1_6"]
    
    loop_n_0 = {}    
    loop_n_1.clear() #empty the previous dictionary
    loop_n_2.clear()
    loop_n_3.clear() 
    loop_n_4.clear()
    
    
    ###############################
    ######### TRAINING 2 #########
    ##############################
    
    df_training_2 = df_exp_fil_trials.loc[df_tr['repeat_training_loop1b.thisRepN'] >= 0]
    df_training_2.dropna(how='all', axis=1)
    Total_resp_Seq_2 = df_training_2["resp_total_corr"].sum()
    
    for iterations_2 in range(0,6):
        df_training_2_loop = df_exp_fil_trials.loc[df_tr['repeat_training_loop1b.thisRepN'] == iterations_2]
        Total_resp_Seq_loop = df_training_2_loop["resp_total_corr"].sum()
        
        loop_n_0[iterations_2] = df_training_2_loop["resp_total_corr"].sum()
       
        df_tr_2_spec_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "spec")]
        df_tr_2_sub_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "subrule")]
        df_tr_2_rule_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "rule")]
        df_tr_2_gen_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "general")]
          
        loop_n_1[iterations_2] = df_tr_2_spec_loop["resp_total_corr"].sum() #loop with its response value
        loop_n_2[iterations_2] = df_tr_2_sub_loop["resp_total_corr"].sum()
        loop_n_3[iterations_2] = df_tr_2_rule_loop["resp_total_corr"].sum()
        loop_n_4[iterations_2] = df_tr_2_gen_loop["resp_total_corr"].sum()
    
             
    df_corr_Seq_per_loop_b = pd.DataFrame.from_dict(loop_n_0,  orient='index')
    df_corr_Seq_per_loop_b = df_corr_Seq_per_loop_b.transpose()
    df_corr_Seq_per_loop_b.columns = ["corr_Sb_1", "corr_Sb_2", "corr_Sb_3", "corr_Sb_4", "corr_Sb_5", "corr_Sb_6"]    
   
    df_corr_Sb_spec_per_loop = pd.DataFrame.from_dict(loop_n_1,  orient='index')
    df_corr_Sb_spec_per_loop = df_corr_Sb_spec_per_loop.transpose()
    df_corr_Sb_spec_per_loop.columns = ["Corr_Sb_spec1_1", "Corr_Sb_spec1_2", "Corr_Sb_spec1_3", "Corr_Sb_spec1_4", "Corr_Sb_spec1_5", "Corr_Sb_spec1_6"]
    
    df_corr_Sb_sub_per_loop = pd.DataFrame.from_dict(loop_n_2,  orient='index')
    df_corr_Sb_sub_per_loop = df_corr_Sb_sub_per_loop.transpose()
    df_corr_Sb_sub_per_loop.columns = ["Corr_Sb_sub1_1", "Corr_Sb_sub1_2", "Corr_Sb_sub1_3", "Corr_Sb_sub1_4", "Corr_Sb_sub1_5", "Corr_Sb_sub1_6"]
    
    df_corr_Sb_rule_per_loop = pd.DataFrame.from_dict(loop_n_3,  orient='index')
    df_corr_Sb_rule_per_loop = df_corr_Sb_rule_per_loop.transpose()
    df_corr_Sb_rule_per_loop.columns = ["Corr_Sb_rule1_1", "Corr_Sb_rule1_2", "Corr_Sb_rule1_3", "Corr_Sb_rule1_4", "Corr_Sb_rule1_5", "Corr_Sb_rule1_6"]
    
    df_corr_Sb_gen_per_loop = pd.DataFrame.from_dict(loop_n_4,  orient='index')
    df_corr_Sb_gen_per_loop = df_corr_Sb_gen_per_loop.transpose()
    df_corr_Sb_gen_per_loop.columns = ["Corr_Sb_gen1_1", "Corr_Sb_gen1_2", "Corr_Sb_gen1_3", "Corr_Sb_gen1_4", "Corr_Sb_gen1_5", "Corr_Sb_gen1_6"]
   
    print(df_corr_Sb_rule_per_loop)
    print(df_corr_Sb_gen_per_loop)
   
    
    loop_n_1.clear() #empty the previous dictionary
    loop_n_2.clear()
    loop_n_3.clear() #empty the previous dictionary
    loop_n_4.clear()    
    
    #Response Time RT TR2
    #clean the dataset and remove filler and incorrect trials
    df_training_2 = df_training_2.drop(df_training_2[df_training_2['trial_type'] == "filler"].index)
    df_training_2 = df_training_2.drop(df_training_2[df_training_2['resp_total_corr'] == 0].index)
    
    df_tr_2_spec_tot = df_training_2.loc[(df_training_2['conditions'] == "spec")]
    RT_Spec_Tot_Tr2 = np.mean(df_tr_2_spec_tot["resp_total_time"])
    
    df_tr_2_sub_tot = df_training_2.loc[(df_training_2['conditions'] == "subrule")]
    RT_Sub_Tot_Tr2 = np.mean(df_tr_2_sub_tot["resp_total_time"])

    df_tr_2_rule_tot = df_training_2.loc[(df_training_2['conditions'] == "rule")]
    RT_Rule_Tot_Tr2 = np.mean(df_tr_2_rule_tot["resp_total_time"])

    df_tr_2_gen_tot = df_training_2.loc[(df_training_2['conditions'] == "general")]
    RT_Gen_Tot_Tr2 = np.mean(df_tr_2_gen_tot["resp_total_time"])

    # RT for each loop TR2
    for iterations_5 in range(0,6):
        df_training_2_loop = df_training_2.loc[df_tr['repeat_training_loop1b.thisRepN'] == iterations_5]
        df_tr_2_spec_loop = df_training_2_loop.loc[(df_training_2['conditions'] == "spec")]
        df_tr_2_sub_loop = df_training_2_loop.loc[(df_training_2['conditions'] == "subrule")]
        df_tr_2_rule_loop = df_training_2_loop.loc[(df_training_2['conditions'] == "rule")]
        df_tr_2_gen_loop = df_training_2_loop.loc[(df_training_2['conditions'] == "general")]
        
        RT_Spec_loop_Tr2 = np.mean(df_tr_2_spec_loop["resp_total_time"])
        RT_Sub_loop_Tr2 = np.mean(df_tr_2_sub_loop["resp_total_time"])
        RT_Rule_loop_Tr2 = np.mean(df_tr_2_rule_loop["resp_total_time"])
        RT_Gen_loop_Tr2 = np.mean(df_tr_2_gen_loop["resp_total_time"])
        
        loop_n_1[iterations_5] = np.mean(df_tr_2_spec_loop["resp_total_time"]) #loop with its response value
        loop_n_2[iterations_5] = np.mean(df_tr_2_sub_loop["resp_total_time"])
        loop_n_3[iterations_5] = np.mean(df_tr_2_rule_loop["resp_total_time"]) #loop with its response value
        loop_n_4[iterations_5] = np.mean(df_tr_2_gen_loop["resp_total_time"])
    
    df_RT2_spec_per_loop = pd.DataFrame.from_dict(loop_n_1,  orient='index')
    df_RT2_spec_per_loop = df_RT2_spec_per_loop.transpose()
    df_RT2_spec_per_loop.columns = ["RT_spec2_1", "RT_spec2_2", "RT_spec2_3", "RT_spec2_4", "RT_spec2_5", "RT_spec2_6"]
 
    df_RT2_sub_per_loop = pd.DataFrame.from_dict(loop_n_2,  orient='index')
    df_RT2_sub_per_loop = df_RT2_sub_per_loop.transpose()
    df_RT2_sub_per_loop.columns = ["RT_sub2_1", "RT_sub2_2", "RT_sub2_3", "RT_sub2_4", "RT_sub2_5", "RT_sub2_6"]
    
    df_RT2_rule_per_loop = pd.DataFrame.from_dict(loop_n_3,  orient='index')
    df_RT2_rule_per_loop = df_RT2_rule_per_loop.transpose()
    df_RT2_rule_per_loop.columns = ["RT_rule2_1", "RT_rule2_2", "RT_rule2_3", "RT_rule2_4", "RT_rule2_5", "RT_rule2_6"]
    
    df_RT2_gen_per_loop = pd.DataFrame.from_dict(loop_n_4,  orient='index')
    df_RT2_gen_per_loop = df_RT2_gen_per_loop.transpose()
    df_RT2_gen_per_loop.columns = ["RT_gen2_1", "RT_gen2_2", "RT_gen2_3", "RT_gen2_4", "RT_gen2_5", "RT_gen2_6"]
     
    loop_n_1.clear()
    loop_n_2.clear()
    loop_n_3.clear() 
    loop_n_4.clear()

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
    for iterations_6 in range(0,6):
        df_training_2_loop = df_training_2.loc[df_tr['repeat_training_loop1b.thisRepN'] == iterations_6]
        df_ot_2_spec_loop = df_training_2_loop.loc[(df_training_2['conditions'] == "spec")]
        df_ot_2_sub_loop = df_training_2_loop.loc[(df_training_2['conditions'] == "subrule")]
        df_ot_2_rule_loop = df_training_2_loop.loc[(df_training_2['conditions'] == "rule")]
        df_ot_2_gen_loop = df_training_2_loop.loc[(df_training_2['conditions'] == "general")]
        
        OT_Spec_loop_Tr2 = np.mean(df_ot_2_spec_loop["resp1b.rt"])
        OT_Sub_loop_Tr2 = np.mean(df_ot_2_sub_loop["resp1b.rt"])
        OT_Rule_loop_Tr2 = np.mean(df_ot_2_rule_loop["resp1b.rt"])
        OT_Gen_loop_Tr2 = np.mean(df_ot_2_gen_loop["resp1b.rt"])
        
        loop_n_1[iterations_6] = np.mean(df_ot_2_spec_loop["resp1b.rt"]) #loop with its response value
        loop_n_2[iterations_6] = np.mean(df_ot_2_sub_loop["resp1b.rt"])
        loop_n_3[iterations_6] = np.mean(df_ot_2_rule_loop["resp1b.rt"])
        loop_n_4[iterations_6] = np.mean(df_ot_2_gen_loop["resp1b.rt"])
    
    df_OT2_spec_per_loop = pd.DataFrame.from_dict(loop_n_1,  orient='index')
    df_OT2_spec_per_loop = df_OT2_spec_per_loop.transpose()
    df_OT2_spec_per_loop.columns = ["OT_spec2_1", "OT_spec2_2", "OT_spec2_3", "OT_spec2_4", "OT_spec2_5", "OT_spec2_6"]
    
    df_OT2_sub_per_loop = pd.DataFrame.from_dict(loop_n_2,  orient='index')
    df_OT2_sub_per_loop = df_OT2_sub_per_loop.transpose()
    df_OT2_sub_per_loop.columns = ["OT_sub2_1", "OT_sub2_2", "OT_sub2_3", "OT_sub2_4", "OT_sub2_5", "OT_sub2_6"]
    
    df_OT2_rule_per_loop = pd.DataFrame.from_dict(loop_n_3,  orient='index')
    df_OT2_rule_per_loop = df_OT2_rule_per_loop.transpose()
    df_OT2_rule_per_loop.columns = ["OT_rule2_1", "OT_rule_2", "OT_rule_3", "OT_rule_4", "OT_rule_5", "OT_rule2_6"]
    
    df_OT2_gen_per_loop = pd.DataFrame.from_dict(loop_n_4,  orient='index')
    df_OT2_gen_per_loop = df_OT2_gen_per_loop.transpose()
    df_OT2_gen_per_loop.columns = ["OT_gen2_1", "OT_gen2_2", "OT_gen2_3", "OT_gen2_4", "OT_gen2_5", "OT_gen2_6"]
     
    loop_n_1.clear() 
    loop_n_2.clear()
    loop_n_3.clear() 
    loop_n_4.clear()
    
    result = pd.concat([df_corr_R_spec_per_loop, df_corr_R_sub_per_loop, df_corr_R_rule_per_loop, 
                        df_corr_R_gen_per_loop, df_corr_S_spec_per_loop, df_corr_S_sub_per_loop, 
                        df_corr_S_rule_per_loop ,df_corr_S_gen_per_loop, 
                        df_RT_spec_per_loop, df_RT_sub_per_loop, df_RT_rule_per_loop,
                        df_RT_gen_per_loop, df_OT_spec_per_loop, df_OT_sub_per_loop, 
                        df_OT_rule_per_loop,df_OT_gen_per_loop, df_RT2_spec_per_loop, 
                        df_RT2_sub_per_loop, df_RT2_rule_per_loop,df_RT2_gen_per_loop, 
                        df_OT2_spec_per_loop, df_OT2_sub_per_loop, df_OT2_rule_per_loop,
                        df_OT2_gen_per_loop, df_corr_Sb_spec_per_loop, df_corr_Sb_sub_per_loop, 
                        df_corr_Sb_rule_per_loop, df_corr_Sb_gen_per_loop], axis=1, sort=False)
    
    result["Subj"] = participant_number

    result["Corr_R_Tot"] = Corr_R_Tot
    result["Corr_S1_Tot"] = Corr_S_Tot

    result["Total_S2_Tot"] = Total_resp_Seq_2

    result["RT_Spec_Tot_Tr1"] = RT_Spec_Tot_Tr1
    result["RT_Sub_Tot_Tr1"] = RT_Sub_Tot_Tr1
    result["RT_Rule_Tot_Tr1"] = RT_Rule_Tot_Tr1
    result["RT_Gen_Tot_Tr1"] = RT_Gen_Tot_Tr1
    
    result["OT_Spec_Tot_Tr1"] = OT_Spec_Tot_Tr1
    result["OT_Sub_Tot_Tr1"] = OT_Sub_Tot_Tr1
    result["OT_Rule_Tot_Tr1"] = OT_Rule_Tot_Tr1
    result["OT_Gen_Tot_Tr1"] = OT_Gen_Tot_Tr1        

    result["RT_Spec_Tot_Tr2"] = RT_Spec_Tot_Tr2
    result["RT_Sub_Tot_Tr2"] = RT_Sub_Tot_Tr2
    result["RT_Rule_Tot_Tr2"] = RT_Rule_Tot_Tr2
    result["RT_Gen_Tot_Tr2"] = RT_Gen_Tot_Tr2
    
    result["OT_Spec_Tot_Tr2"] = OT_Spec_Tot_Tr2
    result["OT_Sub_Tot_Tr2"] = OT_Sub_Tot_Tr2
    result["OT_Rule_Tot_Tr2"] = OT_Rule_Tot_Tr2
    result["OT_Gen_Tot_Tr2"] = OT_Gen_Tot_Tr2     
    
    df_result_tr = df_result_tr.append(result)
    
     
columns_names = list(df_result_tr.columns.values.tolist())

#with open('columns_names.txt', 'w') as f:
#    for columns_name in columns_names:
#        print >> f, columns_name

df_result_tr.to_csv("results.csv", index = False, header=True)
