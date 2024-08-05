# Sequence-based Detection of Protein Interaction Sites
## - data_collection
This folder consists of the following files:
-supplementary_table_S2_liu_et_al.xlsx
-PDBIDs_original.xlsx
-PDBIDs_training.xlsx
-PDBIDs_junction.xlsx
The list of PDBIDs was collected from Supplementary Table 2 of a paper by Liu et al. titled 'Hot spot prediction in protein-protein interactions by an ensemble system'. These were copied into the file supplementary_table_S2_liu_et_al.xlsx. Using supplementary_data_formatting.py, we were able to obtain 213 unique PDBIDs with protein chains (list can be found in PDBIDs_original.xlsx). These were split into two groups, as shown in the Python notebook file supplementary_data_formatting.ipynb- 100 for the training dataset (list can be found in PDBIDs_training.xlax) and 113 for the junction propensity dataset (list can be found in PDBIDs_junction.xlsx).
## - training_data
This folder consists of PBDIDs collected from various databases. Their fasta sequences were obtained from PDB. The features were collected from the codes in feature_collection folder. Several ML models were run on this dataset to obtain maximum accuracy.
## -junction_propensity
This folder contains the csv dataset used to collect secondary structure propensity information.
## -viral_data
This folder consists of our final dataset for external validation. We wish to predict interaction sites for Influenza A viral proteins.
## -ML
This folder consists of several ML models that were used to predict protein interaction sites using data from PDBsum. 
