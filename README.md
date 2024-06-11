# Sequence-based Detection of Protein Interaction Sites
## - feature_collection
This folder consists of Python codes to obtain the propensity of 8 amino acids (W,Y,R,L,V,T), secondary acid propensity (including junctions) and RelSASA (obtained from NetSurfP). These features were used to compile our training dataset and viral dataset. The information to calculate secondary structure propensity is mentioned in the junction_propensity folder.
## - training_data
This folder consists of PBDIDs collected from various databases. Their fasta sequences were obtained from PDB. The features were collected from the codes in feature_collection folder. Several ML models were run on this dataset to obtain maximum accuracy.
## -junction_propensity
This folder contains the csv dataset used to collect secondary structure propensity information.
## -viral_data
This folder consists of our final dataset for external validation. We wish to predict interaction sites for Influenza A viral proteins.
## -ML
This folder consists of several ML models that were used to predict protein interaction sites using data from PDBsum. 
