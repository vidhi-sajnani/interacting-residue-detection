# Sequence-based Detection of Protein Interaction Sites

## - data_collection
This folder consists of the following files:
- supplementary_table_S2_liu_et_al.xlsx
- PDBIDs_original.xlsx
- PDBIDs_training.xlsx
- PDBIDs_junction.xlsx
- supplementary_data_formatting.py
- supplementary_data_formatting.ipynb
The list of PDBIDs was collected from Supplementary Table 2 of a paper by Liu et al. titled 'Hot spot prediction in protein-protein interactions by an ensemble system'. These were copied into the file supplementary_table_S2_liu_et_al.xlsx. Using supplementary_data_formatting.py, we were able to obtain 213 unique PDBIDs with protein chains (list can be found in PDBIDs_original.xlsx). These were split into two groups, as shown in the Python notebook file supplementary_data_formatting.ipynb- 100 for the training dataset (list can be found in PDBIDs_training.xlsx) and 113 for the junction propensity dataset (list can be found in PDBIDs_junction.xlsx).

## - feature_collection
This folder consists of the following files:
- supplementary_table_S2_liu_et_al.xlsx

## - training_data
This folder consists of the following files:
- final_training.csv
- training_fasta_seqs.fasta
- NetSurfP_RelSASA_output.csv
The file training_fasta_seqs.fasta has the fasta sequences of all the PDBID chains chosen for the training dataset. NetSurfP_RelSASA_output.csv consists of the RelSASA values for these protein chains found using NetSurfP 3.0 via web server (https://services.healthtech.dtu.dk/services/NetSurfP-3.0/). The file final_training.csv consists of the training dataset with all the features collected using the code in the feature_collection folder. It also has the label values from PDBSum (indicating if the residue is interacting or not), leaving us with only 70 unique protein chains in the dataset instead of 100.


## -junction_propensity
This folder consists of the following files:
- supplementary_table_S2_liu_et_al.xlsx


## -test_data
This folder consists of the following files:
- supplementary_table_S2_liu_et_al.xlsx


## -supervised_ML
This folder consists of the following files:
- supplementary_table_S2_liu_et_al.xlsx

## -unsupervised_ML
This folder consists of the following files:
- supplementary_table_S2_liu_et_al.xlsx
   
