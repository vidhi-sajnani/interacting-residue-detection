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

## - feature_and_label_collection
This folder consists of the following files:
- inputs
- PDBSum_labels.csv
- junctions_and_secondary_structure_values.ipynb
- matchingPDBSUM.ipynb
- obtaining_labels_list_PDBSum.sh
- propensities_hydrophobicity.ipynb
- newrunpython.py
- awkcommands.txt

Once the fasta sequences for all proteins are obtained from RCSB PDB, we use S4PRED to obtain the secondary structures (https://github.com/psipred/s4pred). The runpython.py code was changed, and the edited version is now in this folder as newrunpython.py. To format the output obtained, certain awk commands (awkcommands.txt) were run to finally get the required output consisting of the PDBIDs, amino acids and secondary structures of all the proteins in vertical format as a .csv file. Using this file as input, we run the code junctions_and_secondary_structure_values.ipynb and propensities_hydrophobicity.ipynb to obtain the junctions, secondary structure propensities and amino acid propensities of all the proteins. We do this step for the training and test datasets present in their respective folders. We also obtain the secondary structures, including junctions for the junction propensity dataset present in its respective folder. 

The final step for the training dataset would be to obtain the Label, ie, whether the given residue is interacting or not (1 or 0). The data for interacting sites was obtained from PDBsum (https://www.ebi.ac.uk/thornton-srv/databases/pdbsum/). The interaction information for each protein chain in the training dataset was manually copied and pasted into .txt files, which are all present in the inputs folder. This consisted of quite a lot of duplicate interaction sites in an odd format. To convert it to a .csv file with only unique interaction sites, we ran obtaining_labels_list_PDBSum.sh. The output obtained was PDBSum_labels.csv which has the list of all the PDBIDs, chains and their interacting residues. Using this as a checklist, we wrote a Python code matchingPDBSUM.ipynb that would check mark 1 in the matching column if the PDBID, chain and residue number were identical in the PDBSum_labels.csv checklist and training csv file. This gave us the output of the training dataset with an additional label column which indicated if the given residue was interacting or not.

## - training_data
This folder consists of the following files:
- final_training.csv
- training_fasta_seqs.fasta
- NetSurfP_RelSASA_output.csv

The file training_fasta_seqs.fasta has the fasta sequences of all the PDBID chains chosen for the training dataset. These were collected from RCSB PDB (https://www.rcsb.org/downloads/fasta). NetSurfP_RelSASA_output.csv consists of the RelSASA values for these protein chains found using NetSurfP 3.0 via a web server (https://services.healthtech.dtu.dk/services/NetSurfP-3.0/). The file final_training.csv consists of the training dataset with all the features collected using the code in the feature_collection folder. It also has the label values from PDBSum (indicating if the residue is interacting or not), leaving us with only 70 unique protein chains in the dataset instead of 100.


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
   
