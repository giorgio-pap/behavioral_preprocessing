import pandas as pd
import os
import re

data_folder_training = "/home/raid2/papitto/Desktop/PsychoPy/MRep_2020_backup/MRep_training_backup/data/"
data_folder_test = "/home/raid2/papitto/Desktop/PsychoPy/MRep_2020_backup/MRep_test_backup/data/"

#read all the files in a folders and creat a list
filenames = os.listdir(data_folder_training)

#create a list only with files UPDATED
UPDATED_files = []

for filename in filenames:
    if filename.endswith("UPDATED.csv"):
        UPDATED_files.append(filename)

#extract the number of the participants
pattern = re.compile(r'(\d*)_Mental')
participant_numbers =  ["; ".join(pattern.findall(item)) for item in UPDATED_files]

#read all the UPDATED files one at the time
for UPDATED_file in UPDATED_files:
    participant_number = [int(s) for s in re.findall(r'(\d*)_Mental', UPDATED_file)]
    str_number = ' '.join(map(str, participant_number)) 
    
    df = pd.read_csv(UPDATED_file,  header=0)
    
    df_exp_fil_trials = df.loc[(df['trial_type'] == "experimental")|(df['trial_type'] == "filler")]
    
    #TRAINING 1
    # ALL LOOPS TOGETHER
    df_training_1 = df_exp_fil_trials.loc[df['repeat_training_loop1.thisRepN'] >= 0]
    df_training_1.dropna(how='all', axis=1)
    Corr_R_Tot = df_training_1["resp_R.corr"].sum() #number of correct relationship answers (incl. fillers) - all loops
    Corr_S_Tot = df_training_1["resp_total_corr"].sum() #number of correct sequences (incl. fillers) - all loops
    
    loop_n_1 = {} #create empty ditctionary
    loop_n_2 = {}
    
    for iterations_1 in range(0,6):
        df_training_1_loop = df_exp_fil_trials.loc[df['repeat_training_loop1.thisRepN'] == iterations_1]
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
      
    # TRAINING 2
    df_training_2 = df_exp_fil_trials.loc[df['repeat_training_loop1b.thisRepN'] >= 0]
    df_training_2.dropna(how='all', axis=1)
    Total_resp_Seq_2 = df_training_2["resp_total_corr"].sum()
    
    #print(Total_resp_Seq_2)
    for iterations_2 in range(0,6):
        df_training_2_loop = df_exp_fil_trials.loc[df['repeat_training_loop1.thisRepN'] == iterations_2]
        df_training_2_loop.dropna(how='all', axis=1)
        Total_resp_Seq_loop = df_training_2_loop["resp_total_corr"].sum()
        loop_n_1[iterations_2] = df_training_2_loop["resp_total_corr"].sum()
             
    df_corr_Seq_per_loop_b = pd.DataFrame.from_dict(loop_n_1,  orient='index')
    df_corr_Seq_per_loop_b = df_corr_Seq_per_loop_b.transpose()
    df_corr_Seq_per_loop_b.columns = ["corr_Sb_1", "corr_Sb_2", "corr_Sb_3", "corr_Sb_4", "corr_Sb_5", "corr_Sb_6"]    
    print(df_corr_Seq_per_loop_b)
    
    loop_n_1.clear() #empty the previous dictionary

    
    df_training_1.to_excel(str_number + "_output_1" + '.xlsx', index=False) 
    df_training_2.to_excel(str_number + "_output_2" + '.xlsx', index=False) 
