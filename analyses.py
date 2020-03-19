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
    
    loop_n_1 = {} #create empty ditctionary
    loop_n_2 = {}
    loop_n_3 = {} #create empty ditctionary
    loop_n_4 = {}
    
    for iterations in range(0,6):
        df_training_1_loop = df_exp_fil_trials.loc[df_tr['repeat_training_loop1.thisRepN'] == iterations]
        Total_resp_R_loop = df_training_1_loop["resp_R.corr"].sum() #number of correct relationship answers (incl. fillers) - all loops
        Total_resp_Seq_loop = df_training_1_loop["resp_total_corr"].sum()
        loop_n_1[iterations] = df_training_1_loop["resp_R.corr"].sum() #loop with its response value
        loop_n_2[iterations] = df_training_1_loop["resp_total_corr"].sum()
        
    df_corr_R_per_loop = pd.DataFrame.from_dict(loop_n_1,  orient='index')
    df_corr_R_per_loop = df_corr_R_per_loop.transpose()
    df_corr_R_per_loop.columns = ["corr_R_1", "corr_R_2", "corr_R_3", "corr_R_4", "corr_R_5", "corr_R_6"]
     
    df_corr_Seq_per_loop = pd.DataFrame.from_dict(loop_n_2,  orient='index')
    df_corr_Seq_per_loop = df_corr_Seq_per_loop.transpose()
    df_corr_R_per_loop.columns = ["corr_Sa_1", "corr_Sa_2", "corr_Sa_3", "corr_Sa_4", "corr_Sa_5", "corr_Sa_6"]    
   
    loop_n_1.clear() #empty the previous dictionary
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
    
    loop_n_1.clear() #empty the dictionary
    loop_n_2.clear()
    loop_n_3.clear()
    loop_n_4.clear()

    for iterations in range(0,6):
        df_training_1_loop = df_training_1.loc[df_tr['repeat_training_loop1.thisRepN'] == iterations]
        df_tr_1_spec_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "spec")] #& (df_training_1['did_get_here'] == 1)]
        df_tr_1_sub_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "subrule")]
        df_tr_1_rule_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "rule")]
        df_tr_1_gen_loop = df_training_1_loop.loc[(df_training_1['conditions'] == "general")]
            

        Corr_S_Spec_loop_Tr1 = df_tr_1_spec_loop["resp_total_corr"].sum()
        Corr_S_Sub_loop_Tr1 = df_tr_1_sub_loop["resp_total_corr"].sum()
        Corr_S_Rule_loop_Tr1 = df_tr_1_rule_loop["resp_total_corr"].sum()
        Corr_S_Gen_loop_Tr1 = df_tr_1_gen_loop["resp_total_corr"].sum()
        
        loop_n_1[iterations] = df_tr_1_spec_loop["resp_total_corr"].sum() #loop with its response value
        loop_n_2[iterations] = df_tr_1_sub_loop["resp_total_corr"].sum()
        loop_n_3[iterations] = df_tr_1_rule_loop["resp_total_corr"].sum()
        loop_n_4[iterations] = df_tr_1_gen_loop["resp_total_corr"].sum()
    

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
    
    df_training_2_wo = df_exp_fil_trials.loc[df_tr['repeat_training_loop1b.thisRepN'] >= 0]
    df_training_2_wo = df_training_2_wo.drop(df_training_1[df_training_1['trial_type'] == "filler"].index)
    Corr_S2_Tot_wo = df_training_2_wo["resp_total_corr"].sum() #number of correct sequences (without fillers) - all loops
    
    
    for iterations in range(0,6):
        df_training_2_loop = df_exp_fil_trials.loc[df_tr['repeat_training_loop1b.thisRepN'] == iterations]
        Total_resp_Seq_loop_te = df_training_2_loop["resp_total_corr"].sum()
        
        loop_n_0[iterations] = df_training_2_loop["resp_total_corr"].sum()
       
        df_tr_2_spec_loop = df_training_2_loop.loc[(df_training_2_loop['conditions'] == "spec")]
        df_tr_2_sub_loop = df_training_2_loop.loc[(df_training_2_loop['conditions'] == "subrule")]
        df_tr_2_rule_loop = df_training_2_loop.loc[(df_training_2_loop['conditions'] == "rule")]
        df_tr_2_gen_loop = df_training_2_loop.loc[(df_training_2_loop['conditions'] == "general")]
          
        loop_n_1[iterations] = df_tr_2_spec_loop["resp_total_corr"].sum() #loop with its response value
        loop_n_2[iterations] = df_tr_2_sub_loop["resp_total_corr"].sum()
        loop_n_3[iterations] = df_tr_2_rule_loop["resp_total_corr"].sum()
        loop_n_4[iterations] = df_tr_2_gen_loop["resp_total_corr"].sum()
    
             
    df_corr_Seq_per_loop_b = pd.DataFrame.from_dict(loop_n_0,  orient='index')
    df_corr_Seq_per_loop_b = df_corr_Seq_per_loop_b.transpose()
    df_corr_Seq_per_loop_b.columns = ["corr_S2_1", "corr_S2_2", "corr_S2_3", "corr_S2_4", "corr_S2_5", "corr_S2_6"]    
   
    df_corr_S2_spec_per_loop = pd.DataFrame.from_dict(loop_n_1,  orient='index')
    df_corr_S2_spec_per_loop = df_corr_S2_spec_per_loop.transpose()
    df_corr_S2_spec_per_loop.columns = ["Corr_S_spec2_1", "Corr_S_spec2_2", "Corr_S_spec2_3", "Corr_S_spec2_4", "Corr_S_spec2_5", "Corr_S_spec2_6"]
    
    df_corr_S2_sub_per_loop = pd.DataFrame.from_dict(loop_n_2,  orient='index')
    df_corr_S2_sub_per_loop = df_corr_S2_sub_per_loop.transpose()
    df_corr_S2_sub_per_loop.columns = ["Corr_S_sub2_1", "Corr_S_sub2_2", "Corr_S_sub2_3", "Corr_S_sub2_4", "Corr_S_sub2_5", "Corr_S_sub2_6"]
    
    df_corr_S2_rule_per_loop = pd.DataFrame.from_dict(loop_n_3,  orient='index')
    df_corr_S2_rule_per_loop = df_corr_S2_rule_per_loop.transpose()
    df_corr_S2_rule_per_loop.columns = ["Corr_S_rule2_1", "Corr_S_rule2_2", "Corr_S_rule2_3", "Corr_S_rule2_4", "Corr_S_rule2_5", "Corr_S_rule2_6"]
    
    df_corr_S2_gen_per_loop = pd.DataFrame.from_dict(loop_n_4,  orient='index')
    df_corr_S2_gen_per_loop = df_corr_S2_gen_per_loop.transpose()
    df_corr_S2_gen_per_loop.columns = ["Corr_S_gen2_1", "Corr_S_gen2_2", "Corr_S_gen2_3", "Corr_S_gen2_4", "Corr_S_gen2_5", "Corr_S_gen2_6"]   
    
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
    
    df_OT2_spec_per_loop = pd.DataFrame.from_dict(loop_n_1,  orient='index')
    df_OT2_spec_per_loop = df_OT2_spec_per_loop.transpose()
    df_OT2_spec_per_loop.columns = ["OT_spec2_1", "OT_spec2_2", "OT_spec2_3", "OT_spec2_4", "OT_spec2_5", "OT_spec2_6"]
    
    df_OT2_sub_per_loop = pd.DataFrame.from_dict(loop_n_2,  orient='index')
    df_OT2_sub_per_loop = df_OT2_sub_per_loop.transpose()
    df_OT2_sub_per_loop.columns = ["OT_sub2_1", "OT_sub2_2", "OT_sub2_3", "OT_sub2_4", "OT_sub2_5", "OT_sub2_6"]
    
    df_OT2_rule_per_loop = pd.DataFrame.from_dict(loop_n_3,  orient='index')
    df_OT2_rule_per_loop = df_OT2_rule_per_loop.transpose()
    df_OT2_rule_per_loop.columns = ["OT_rule2_1", "OT_rule2_2", "OT_rule2_3", "OT_rule2_4", "OT_rule2_5", "OT_rule2_6"]
    
    df_OT2_gen_per_loop = pd.DataFrame.from_dict(loop_n_4,  orient='index')
    df_OT2_gen_per_loop = df_OT2_gen_per_loop.transpose()
    df_OT2_gen_per_loop.columns = ["OT_gen2_1", "OT_gen2_2", "OT_gen2_3", "OT_gen2_4", "OT_gen2_5", "OT_gen2_6"]

    loop_n_0.clear()      
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
                        df_OT2_gen_per_loop, df_corr_S2_spec_per_loop, df_corr_S2_sub_per_loop, 
                        df_corr_S2_rule_per_loop, df_corr_S2_gen_per_loop], axis=1, sort=False)
    
    result["Subj_tr"] = participant_number_tr

    result["Corr_R_Tot"] = Corr_R_Tot
    result["Corr_S1_Tot"] = Corr_S_Tot
    
    result["Corr_R_Tot_wo"] = Corr_R_Tot_wo
    result["Corr_S1_Tot_wo"] = Corr_S_Tot_wo
    
    result["Corr_R_spec1"] = df_corr_R_spec_per_loop.sum(axis=1)
    result["Corr_R_sub1"] = df_corr_R_sub_per_loop.sum(axis=1)
    result["Corr_R_rule1"] = df_corr_R_rule_per_loop.sum(axis=1)
    result["Corr_R_gen1"] = df_corr_R_gen_per_loop.sum(axis=1)

    result["Corr_S_spec1"] = df_corr_S_spec_per_loop.sum(axis=1)
    result["Corr_S_sub1"] = df_corr_S_sub_per_loop.sum(axis=1)
    result["Corr_S_rule1"] = df_corr_S_rule_per_loop.sum(axis=1)
    result["Corr_S_gen1"] = df_corr_S_gen_per_loop.sum(axis=1)    
    
    result["Corr_S2_Tot"] = Total_resp_Seq_2
    result["Corr_S2_Tot_wo"] = Corr_S2_Tot_wo
    
    result["RT_spec1_Tot"] = RT_Spec_Tot_Tr1
    result["RT_sub1_Tot"] = RT_Sub_Tot_Tr1
    result["RT_rule1_Tot"] = RT_Rule_Tot_Tr1
    result["RT_gen1_Tot"] = RT_Gen_Tot_Tr1
    
    result["OT_spec1_Tot"] = OT_Spec_Tot_Tr1
    result["OT_sub1_Tot"] = OT_Sub_Tot_Tr1
    result["OT_rule1_Tot"] = OT_Rule_Tot_Tr1
    result["OT_gen1_Tot"] = OT_Gen_Tot_Tr1        


    result["RT_spec2_Tot"] = RT_Spec_Tot_Tr2
    result["RT_sub2_Tot"] = RT_Sub_Tot_Tr2
    result["RT_rule2_Tot"] = RT_Rule_Tot_Tr2
    result["RT_gen2_Tot"] = RT_Gen_Tot_Tr2
    
    result["OT_spec2_Tot"] = OT_Spec_Tot_Tr2
    result["OT_sub2_Tot"] = OT_Sub_Tot_Tr2
    result["OT_rule2_Tot"] = OT_Rule_Tot_Tr2
    result["OT_gen2_Tot"] = OT_Gen_Tot_Tr2     
    
    result["Corr_S_spec2"] = df_corr_S2_spec_per_loop.sum(axis=1)
    result["Corr_S_sub2"] = df_corr_S2_sub_per_loop.sum(axis=1)
    result["Corr_S_rule2"] = df_corr_S2_rule_per_loop.sum(axis=1)
    result["Corr_S_gen2"] = df_corr_S2_gen_per_loop.sum(axis=1)    
    
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
    Corr_ST_Tot = df_te["resp_total_corr"].sum()      

    df_te_wo = df_te.loc[df_te['file_n'] >= 1]
    df_te_wo = df_te_wo.drop(df_te_wo[df_te_wo['trial_type'] == "filler"].index)
    Corr_ST_Tot_wo = df_te_wo["resp_total_corr"].sum() 
    
    df_te = df_te.loc[df_te['file_n'] >= 1]

    for iterations in range(1,7):
        df_te_loop = df_te.loc[df_te['file_n'] == iterations]
        
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
    
         
    df_corr_Seq_per_loop_te = pd.DataFrame.from_dict(loop_n_0,  orient='index')
    df_corr_Seq_per_loop_te = df_corr_Seq_per_loop_te.transpose()
    df_corr_Seq_per_loop_te.columns = ["corr_ST_1", "corr_ST_2", "corr_ST_3", "corr_ST_4", "corr_ST_5", "corr_ST_6"]    
   
    df_corr_ST_spec_per_loop_te = pd.DataFrame.from_dict(loop_n_1,  orient='index')
    df_corr_ST_spec_per_loop_te = df_corr_ST_spec_per_loop_te.transpose()
    df_corr_ST_spec_per_loop_te.columns = ["Corr_ST_spec_1", "Corr_ST_spec_2", "Corr_ST_spec_3", "Corr_ST_spec_4", "Corr_ST_spec_5", "Corr_ST_spec_6"]
    
    df_corr_ST_sub_per_loop_te = pd.DataFrame.from_dict(loop_n_2,  orient='index')
    df_corr_ST_sub_per_loop_te = df_corr_ST_sub_per_loop_te.transpose()
    df_corr_ST_sub_per_loop_te.columns = ["Corr_ST_sub_1", "Corr_ST_sub_2", "Corr_ST_sub_3", "Corr_ST_sub_4", "Corr_ST_sub_5", "Corr_ST_sub_6"]
    
    df_corr_ST_rule_per_loop_te = pd.DataFrame.from_dict(loop_n_3,  orient='index')
    df_corr_ST_rule_per_loop_te = df_corr_ST_rule_per_loop_te.transpose()
    df_corr_ST_rule_per_loop_te.columns = ["Corr_ST_rule_1", "Corr_ST_rule_2", "Corr_ST_rule_3", "Corr_ST_rule_4", "Corr_ST_rule_5", "Corr_ST_rule_6"]
    
    df_corr_ST_gen_per_loop_te = pd.DataFrame.from_dict(loop_n_4,  orient='index')
    df_corr_ST_gen_per_loop_te = df_corr_ST_gen_per_loop_te.transpose()
    df_corr_ST_gen_per_loop_te.columns = ["Corr_ST_gen_1", "Corr_ST_gen_2", "Corr_ST_gen_3", "Corr_ST_gen_4", "Corr_ST_gen_5", "Corr_ST_gen_6"]   
    
    loop_n_0.clear()
    loop_n_1.clear() #empty the previous dictionary
    loop_n_2.clear()
    loop_n_3.clear() 
    loop_n_4.clear()    
    
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
    
    df_RT_spec_per_loop_te = pd.DataFrame.from_dict(loop_n_1,  orient='index')
    df_RT_spec_per_loop_te = df_RT_spec_per_loop_te.transpose()
    df_RT_spec_per_loop_te.columns = ["RT_specT_1", "RT_specT_2", "RT_specT_3", "RT_specT_4", "RT_specT_5", "RT_specT_6"]
 
    df_RT_sub_per_loop_te = pd.DataFrame.from_dict(loop_n_2,  orient='index')
    df_RT_sub_per_loop_te = df_RT_sub_per_loop_te.transpose()
    df_RT_sub_per_loop_te.columns = ["RT_subT_1", "RT_subT_2", "RT_subT_3", "RT_subT_4", "RT_subT_5", "RT_subT_6"]
    
    df_RT_rule_per_loop_te = pd.DataFrame.from_dict(loop_n_3,  orient='index')
    df_RT_rule_per_loop_te = df_RT_rule_per_loop_te.transpose()
    df_RT_rule_per_loop_te.columns = ["RT_ruleT_1", "RT_ruleT_2", "RT_ruleT_3", "RT_ruleT_4", "RT_ruleT_5", "RT_ruleT_6"]
    
    df_RT_gen_per_loop_te = pd.DataFrame.from_dict(loop_n_4,  orient='index')
    df_RT_gen_per_loop_te = df_RT_gen_per_loop_te.transpose()
    df_RT_gen_per_loop_te.columns = ["RT_genT_1", "RT_genT_2", "RT_genT_3", "RT_genT_4", "RT_genT_5", "RT_genT_6"]
     
    loop_n_1.clear()
    loop_n_2.clear()
    loop_n_3.clear() 
    loop_n_4.clear()
    
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
    
    df_OT_spec_per_loop_te = pd.DataFrame.from_dict(loop_n_1,  orient='index')
    df_OT_spec_per_loop_te = df_OT_spec_per_loop_te.transpose()
    df_OT_spec_per_loop_te.columns = ["OT_specT_1", "OT_specT_2", "OT_specT_3", "OT_specT_4", "OT_specT_5", "OT_specT_6"]
    
    df_OT_sub_per_loop_te = pd.DataFrame.from_dict(loop_n_2,  orient='index')
    df_OT_sub_per_loop_te = df_OT_sub_per_loop_te.transpose()
    df_OT_sub_per_loop_te.columns = ["OT_subT_1", "OT_subT_2", "OT_subT_3", "OT_subT_4", "OT_subT_5", "OT_subT_6"]
    
    df_OT_rule_per_loop_te = pd.DataFrame.from_dict(loop_n_3,  orient='index')
    df_OT_rule_per_loop_te = df_OT_rule_per_loop_te.transpose()
    df_OT_rule_per_loop_te.columns = ["OT_ruleT_1", "OT_ruleT_2", "OT_ruleT_3", "OT_ruleT_4", "OT_ruleT_5", "OT_ruleT_6"]
    
    df_OT_gen_per_loop_te = pd.DataFrame.from_dict(loop_n_4,  orient='index')
    df_OT_gen_per_loop_te = df_OT_gen_per_loop_te.transpose()
    df_OT_gen_per_loop_te.columns = ["OT_genT_1", "OT_genT_2", "OT_genT_3", "OT_genT_4", "OT_genT_5", "OT_genT_6"]
     
    loop_n_1.clear() 
    loop_n_2.clear()
    loop_n_3.clear() 
    loop_n_4.clear()    
    
    result_te = pd.concat([df_corr_Seq_per_loop_te, df_corr_ST_spec_per_loop_te, df_corr_ST_sub_per_loop_te, df_corr_ST_rule_per_loop_te, df_corr_ST_gen_per_loop_te,
                          df_RT_spec_per_loop_te, df_RT_sub_per_loop_te, df_RT_rule_per_loop_te, df_RT_gen_per_loop_te,
                          df_OT_spec_per_loop_te, df_OT_sub_per_loop_te, df_OT_rule_per_loop_te, df_OT_gen_per_loop_te], axis=1, sort=False)

    result_te["Corr_ST_Tot"] = Corr_ST_Tot
    result_te["Corr_ST_Tot_wo"] = Corr_ST_Tot_wo

    result_te["Corr_ST_spec"] = df_corr_ST_spec_per_loop_te.sum(axis=1)
    result_te["Corr_ST_sub"] = df_corr_ST_sub_per_loop_te.sum(axis=1)
    result_te["Corr_ST_rule"] = df_corr_ST_rule_per_loop_te.sum(axis=1)
    result_te["Corr_ST_gen"] = df_corr_ST_gen_per_loop_te.sum(axis=1)    
    
    result_te["RT_Spec_Tot_Te"] = RT_Spec_Tot_Te
    result_te["RT_Sub_Tot_Te"] = RT_Sub_Tot_Te
    result_te["RT_Rule_Tot_Te"] = RT_Rule_Tot_Te
    result_te["RT_Gen_Tot_Te"] = RT_Gen_Tot_Te

    result_te["OT_Spec_Tot_Te"] = OT_Spec_Tot_Te
    result_te["OT_Sub_Tot_Te"] = OT_Sub_Tot_Te
    result_te["OT_Rule_Tot_Te"] = OT_Rule_Tot_Te
    result_te["oT_Gen_Tot_Te"] = OT_Gen_Tot_Te
    
    result_te["Subj_te"] = str_number_te
    
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
