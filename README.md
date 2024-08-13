# Sequence-based Detection of Protein Interaction Sites

This pipeline aims to predict protein interaction sites in Influenza A viral proteins using only the amino acid sequence of a protein. The respective folders have been described below:

## - data_collection
This folder consists of the following files:
- supplementary_table_S2_liu_et_al.xlsx
- PDBIDs_original.xlsx
- PDBIDs_training.xlsx
- PDBIDs_junction.xlsx
- supplementary_data_formatting.py
- supplementary_data_formatting.ipynb


The list of PDBIDs was collected from Supplementary Table 2 of a paper by Liu et al. titled 'Hot spot prediction in protein-protein interactions by an ensemble system' (https://bmcsystbiol.biomedcentral.com/articles/10.1186/s12918-018-0665-8). These were copied into the file supplementary_table_S2_liu_et_al.xlsx. Using supplementary_data_formatting.py, we were able to obtain 213 unique PDBIDs with protein chains (list can be found in PDBIDs_original.xlsx). These were split into two groups, as shown in the Python notebook file supplementary_data_formatting.ipynb- 100 for the training dataset (list can be found in PDBIDs_training.xlsx) and 113 for the junction propensity dataset (list can be found in PDBIDs_junction.xlsx).

## -junction_propensity
This folder consists of the following files:
- FINAL_junction.csv
- junction_propensity_dataset.xlsx
- junction_sequences.fasta

The file junction_sequences.fasta consists of the fasta sequences of all the protein chains selected for the junction propensity dataset. We obtained their secondary structures from S4PRED and then ran the codes in the feature_collection folder to obtain the junctions and also the labels from PDBsum (junction_propensity_dataset.xlsx). The process of calculating the secondary structure propensities is described in detail in our paper. The FINAL_junction.csv shows the final calculations of 69 proteins and their interaction sites as secondary structures. From this calculation, we obtained the final junction propensities used in junctions_and_secondary_structure_values.ipynb in the feature_collection folder.

## - feature_collection
This folder consists of the following files:
- inputs
- PDBSum_labels.csv
- junctions_and_secondary_structure_values.ipynb
- junctions_and_secondary_structure_values.py
- matchingPDBSUM.ipynb
- matchingPDBSUM.py
- obtaining_labels_list_PDBSum.sh
- propensities_hydrophobicity.ipynb
- propensities_hydrophobicity.py
- newrun_model.py
- awkcommands.txt
- NetSurfP_RelSASA_output.csv

Once the fasta sequences for all proteins are obtained from RCSB PDB, we use S4PRED to obtain the secondary structures (https://github.com/psipred/s4pred). The runpython.py code was changed, and the edited version is now in this folder as newrunpython.py. To format the output obtained, certain awk commands (awkcommands.txt) were run to finally get the required output consisting of the PDBIDs, amino acids and secondary structures of all the proteins in vertical format as a .csv file. Using this file as input, we run the code junctions_and_secondary_structure_values.ipynb and propensities_hydrophobicity.ipynb to obtain the junctions, secondary structure propensities and amino acid propensities of all the proteins. We do this step for the training and test datasets present in their respective folders. We also obtain the secondary structures, including junctions for the junction propensity dataset present in its respective folder. NetSurfP_RelSASA_output.csv consists of the RelSASA values for these protein chains found using NetSurfP 3.0 via a web server (https://services.healthtech.dtu.dk/services/NetSurfP-3.0/).

The final step for the training dataset would be to obtain the Label, ie, whether the given residue is interacting or not (1 or 0). The data for interacting sites was obtained from PDBsum (https://www.ebi.ac.uk/thornton-srv/databases/pdbsum/). The interaction information for each protein chain in the training and junction propensity dataset was manually copied and pasted into .txt files, which are all present in the inputs folder. This consisted of quite a lot of duplicate interaction sites in an odd format. To convert it to a .csv file with only unique interaction sites, we ran obtaining_labels_list_PDBSum.sh. The output obtained was PDBSum_labels.csv which has the list of all the PDBIDs, chains and their interacting residues. Using this as a checklist, we wrote a Python code matchingPDBSUM.ipynb that would check mark 1 in the matching column if the PDBID, chain and residue number were identical in the PDBSum_labels.csv checklist and training csv file. This gave us the output of the training dataset with an additional label column which indicated if the given residue was interacting or not. The identical process was repeated for the junction propensity dataset.

## - training_data
This folder consists of the following files:
- final_training.csv
- training_fasta_seqs.fasta

The file training_fasta_seqs.fasta has the fasta sequences of all the PDBID chains chosen for the training dataset. These were collected from RCSB PDB (https://www.rcsb.org/downloads/fasta). The file final_training.csv consists of the training dataset with all the features collected using the code in the feature_collection folder. It also has the label values from PDBSum (indicating if the residue is interacting or not) obtained from the code in the feature_collection folder. This is used as the input into the supervised ML models.


## -test_data
This folder consists of the following files:
- test_data.fasta
- viral_test_dataset.xlsx
- external-val.xlsx

The file test_data.fasta consists of the consensus sequences of the 10 Influenza A viral proteins obtained from Lubna et al. (https://www.authorea.com/users/658468/articles/662811-consortium-of-consistent-amino-acid-substitutions-on-influenza-a-h1n1-viral-proteome-emerged-at-specific-stages-of-viral-infection-a-big-data-analysis). This is used as an input to obtain secondary structures, RelSASA and amino acid propensities, as mentioned in the feature_collection folder. Since this is a test dataset, we do not have any Label values. The viral_test_dataset.xlsx file is the file with all the features of the viral proteins. It is used to test out supervised ML models. The file external-val.xlsx consists of the test dataset with Label values from the paper by Lubna et al. (Table 3). This is to measure the performance of our machine learning models.


## -supervised_ML
This folder consists of the following files:
- gradientboosting.ipynb
- logisticregression.ipynb
- naivebayes.ipynb
- randomforest.ipynb
- svm.ipynb
- xgboost.ipynb
- gradientboosting.py
- logisticregression.py
- naivebayes.py
- randomforest.py
- svm.py
- xgboost.py

This folder consists of all the supervised machine learning models named after the respective models used. They all take final_training.csv and viral_test_dataset.xlsx as training and test dataset inputs, respectively. The accuracies and other parameters of the models are displayed in our paper. 

## -unsupervised_ML
This folder consists of the following files:
- external_validation_sequences.fas
- Unsupervised_model_predictions.ipynb
   
