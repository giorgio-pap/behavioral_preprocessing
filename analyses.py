import pandas as pd
import os
import re
import numpy as np 

data_folder_training = "/home/raid2/papitto/Desktop/PsychoPy/MRep_2020_backup/MRep_training_backup/data/"
data_folder_test = "/home/raid2/papitto/Desktop/PsychoPy/MRep_2020_backup/MRep_test_backup/data/"

df_result_tr = pd.DataFrame([]) #empty dataframe for result file
df_result_te = pd.DataFrame([])

#read all the files in a folders and creat a list
filenames_tr = os.listdir(data_folder_training)
filenames_te = os.listdir(data_folder_test)

#create a list only with files UPDATED
UPDATED_files_tr = []
UPDATED_files_te = []

for filename_tr in filenames_tr:
    if filename_tr.endswith("UPDATED.csv"):
        UPDATED_files_tr.append(filename_tr)
for filename_te in filenames_te:
    if filename_te.endswith("UPDATED.csv"):
        UPDATED_files_te.append(filename_te)
        
UPDATED_files_tr = sorted(UPDATED_files_tr)
UPDATED_files_te = sorted(UPDATED_files_te)

#extract the number of the participants
pattern = re.compile(r'(\d*)_Mental')
participant_numbers =  ["; ".join(pattern.findall(item)) for item in UPDATED_files_tr]


#read all the UPDATED files one at the time
for UPDATED_file_tr in UPDATED_files_tr: #UPDATED_file_tr = each participant (number)
    
    participant_number_tr = [int(s) for s in re.findall(r'(\d*)_Mental', UPDATED_file_tr)]
    str_number = ' '.join(map(str, participant_number_tr)) 

    df_tr = pd.read_csv(UPDATED_file_tr,  header=0)
    
    df_exp_fil_trials = df_tr.loc[(df_tr['trial_type'] == "experimental")|(df_tr['trial_type'] == "filler")]
    
    group = df_exp_fil_trials["group"].iloc[0]

    ###############################
    ######### TRAINING 1 #########
    ##############################

    # ALL LOOPS TOGETHER
    df_training_1 = df_exp_fil_trials.loc[df_tr['repeat_training_loop1.thisRepN'] >= 0]
    Corr_R_Tot = df_training_1["resp_R.corr"].sum() #number of correct relationship answers (incl. fillers) - all loops
    Corr_S_Tot = df_training_1["resp_total_corr"].sum() #number of correct sequences (incl. fillers) - all loops
    
    df_training_1_wo = df_exp_fil_trials.loc[df_tr['repeat_training_loop1.thisRepN'] >= 0]
    df_training_1_wo = df_training_1_wo.drop(df_training_1[df_training_1['trial_type'] == "filler"].index)
    Corr_R_Tot_wo = df_training_1_wo["resp_R.corr"].sum() #number of correct relationship answers (without fillers) - all loops
    Corr_S_Tot_wo = df_training_1_wo["resp_total_corr"].sum() #number of correct sequences (without fillers) - all loops
    
    loop_n_0 = {}    
    loop_n_1 = {} 
    loop_n_2 = {}
    loop_n_3 = {} 
    loop_n_4 = {}
    
    for iterations in range(0,6):
        df_training_1_loop = df_exp_fil_trials.loc[df_exp_fil_trials['repeat_training_loop1.thisRepN'] == iterations]
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

    loop_n_1.clear() #empty the previous dictionary
    loop_n_2.clear()
    loop_n_1.clear()
    loop_n_2.clear()
    
    # Corr_R for each loop TR1
    #FILLER TRIALS ARE NOT CONSIDERED ANYMORE
    for iterations in range(0,6):
        df_training_1_loop = df_training_1.loc[df_tr['repeat_training_loop1.thisRepN'] == iterations]
        df_tr_1_spec_loop = df_training_1_loop.loc[(df_training_1_loop['conditions'] == "spec")]
        df_tr_1_sub_loop = df_training_1_loop.loc[(df_training_1_loop['conditions'] == "subrule")]
        df_tr_1_rule_loop = df_training_1_loop.loc[(df_training_1_loop['conditions'] == "rule")]
        df_tr_1_gen_loop = df_training_1_loop.loc[(df_training_1_loop['conditions'] == "general")]
        
        Corr_R_Spec_loop_Tr1 = df_tr_1_spec_loop["resp_R.corr"].sum()
        Corr_R_Sub_loop_Tr1 = df_tr_1_sub_loop["resp_R.corr"].sum()
        Corr_R_Rule_loop_Tr1 = df_tr_1_rule_loop["resp_R.corr"].sum()
        Corr_R_Gen_loop_Tr1 = df_tr_1_gen_loop["resp_R.corr"].sum()      
 
        loop_n_1[iterations] = df_tr_1_spec_loop["resp_R.corr"].sum() #loop with its response value
        loop_n_2[iterations] = df_tr_1_sub_loop["resp_R.corr"].sum()
        loop_n_3[iterations] = df_tr_1_rule_loop["resp_R.corr"].sum()
        loop_n_4[iterations] = df_tr_1_gen_loop["resp_R.corr"].sum()
         
                
    df_Corr_R_spec_per_loop = pd.DataFrame.from_dict(loop_n_1,  orient='index')
    df_Corr_R_spec_per_loop = df_Corr_R_spec_per_loop.transpose()
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
    
    
    for iterations in range(0,6):
        df_training_1_loop = df_training_1.loc[df_tr['repeat_training_loop1.thisRepN'] == iterations]
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
    df_training_1 = df_training_1.drop(df_training_1[df_training_1['resp_total_corr'] == 0].index)
    
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
        df_training_1_loop = df_training_1.loc[df_tr['repeat_training_loop1.thisRepN'] == iterations]
        df_tr_1_spec_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "spec")]
        df_tr_1_sub_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "subrule")]
        df_tr_1_rule_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "rule")]
        df_tr_1_gen_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "general")]
        
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
        df_ot_1_spec_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "spec")]
        df_ot_1_sub_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "subrule")]
        df_ot_1_rule_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "rule")]
        df_ot_1_gen_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "general")]
        
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
    
    df_training_2 = df_exp_fil_trials.loc[df_tr['repeat_training_loop1b.thisRepN'] >= 0]
    df_training_2.dropna(how='all', axis=1)
    Total_resp_Seq_2 = df_training_2["resp_total_corr"].sum()
    
    df_training_2_wo = df_exp_fil_trials.loc[df_tr['repeat_training_loop1b.thisRepN'] >= 0]
    df_training_2_wo = df_training_2_wo.drop(df_training_1[df_training_1['trial_type'] == "filler"].index)
    Corr_S2_Tot_wo = df_training_2_wo["resp_total_corr"].sum() #number of correct sequences (without fillers) - all loops
    
    for iterations in range(0,6):
        df_training_2_loop = df_exp_fil_trials.loc[df_tr['repeat_training_loop1b.thisRepN'] == iterations]
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
    for iterations in range(0,6):
        df_training_2_loop = df_training_2.loc[df_tr['repeat_training_loop1b.thisRepN'] == iterations]
        
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
        df_training_2_loop = df_training_2.loc[df_training_2['repeat_training_loop1b.thisRepN'] == iterations]
        
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
    
    df_result_tr = df_result_tr.append(result)


###############################
############ TEST ############
##############################
    
pattern = re.compile(r'(\d*)_Mental')
participant_numbers =  ["; ".join(pattern.findall(item)) for item in UPDATED_files_te]

for UPDATED_file_te in UPDATED_files_te: #UPDATED_file_te = each participant (number)
    
    participant_number_te = [int(s) for s in re.findall(r'(\d*)_Mental', UPDATED_file_te)]
    str_number_te = ' '.join(map(str, participant_number_te))
   
    
    df_te = pd.read_csv(data_folder_test + UPDATED_file_te,  header=0)
    df_te = df_te.loc[(df_te['trial_type'] == "experimental")|(df_te['trial_type'] == "filler")]
    Corr_S3_Tot = df_te["resp_total_corr"].sum()      

    df_te_wo = df_te.loc[df_te['file_n'] >= 1]
    df_te_wo = df_te_wo.drop(df_te_wo[df_te_wo['trial_type'] == "filler"].index)
    Corr_S3_Tot_wo = df_te_wo["resp_total_corr"].sum() 
    
    df_te = df_te.loc[df_te['file_n'] >= 1]
    df_te_wo = df_te.loc[(df_te['file_n'] >= 1) & (df_te['trial_type'] == "experimental")]

    for iterations in range(1,7):
        df_te_loop = df_te.loc[df_te['file_n'] == iterations]
        df_te_loop_wo = df_te_wo.loc[df_te['file_n'] == iterations]
        
        Total_resp_Seq_loop_te = df_te_loop["resp_total_corr"].sum()

        loop_n_0[iterations] = df_te_loop["resp_total_corr"].sum()
       
        df_te_spec_loop = df_te_loop.loc[(df_te_loop['conditions'] == "spec")]
        df_te_sub_loop = df_te_loop.loc[(df_te_loop['conditions'] == "subrule")]
        df_te_rule_loop = df_te_loop.loc[(df_te_loop['conditions'] == "rule")]
        df_te_gen_loop = df_te_loop.loc[(df_te_loop['conditions'] == "general")]
          
        loop_n_1[iterations] = df_te_spec_loop["resp_total_corr"].sum() #loop with its response value
        loop_n_2[iterations] = df_te_sub_loop["resp_total_corr"].sum()
        loop_n_3[iterations] = df_te_rule_loop["resp_total_corr"].sum()
        loop_n_4[iterations] = df_te_gen_loop["resp_total_corr"].sum()
        loop_n_5[iterations] = df_te_loop_wo["resp_total_corr"].sum()
    
         
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
    df_Corr_S3_per_loop_wo.columns = ["Corr_S3_1", "Corr_S3_2", "Corr_S3_3", "Corr_S3_4", "Corr_S3_5", "Corr_S3_6"]    

    loop_n_0.clear()
    loop_n_1.clear() #empty the previous dictionary
    loop_n_2.clear()
    loop_n_3.clear() 
    loop_n_4.clear()    
    loop_n_5.clear()
    
    #Response Time RT TEST
    #clean the dataset and remove filler and incorrect trials
    df_te = df_te.drop(df_te[df_te['trial_type'] == "filler"].index)
    df_te = df_te.drop(df_te[df_te['resp_total_corr'] == 0].index)
    
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
        
    result_te = pd.concat([df_Corr_S3_per_loop, df_Corr_S3_spec_per_loop, df_Corr_S3_sub_per_loop, df_Corr_S3_rule_per_loop, df_Corr_S3_gen_per_loop,
                          df_RT3_spec_per_loop, df_RT3_sub_per_loop, df_RT3_rule_per_loop, df_RT3_gen_per_loop,
                          df_OT3_spec_per_loop, df_OT3_sub_per_loop, df_OT3_rule_per_loop, df_OT3_gen_per_loop, 
                          df_Corr_S3_per_loop_wo, df_RT3_per_loop, df_RT3_spec_per_loop, df_OT3_per_loop], axis=1, sort=False)

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
    
result_total = pd.concat([df_result_tr, df_result_te], axis=1, sort=False)

columns_names = list(result_total.columns.values.tolist())

with open('columns_names.txt', 'w') as f:
    for columns_name in columns_names:
        print >> f, columns_name

result_total.to_csv("results.csv", index = False, header=True)
