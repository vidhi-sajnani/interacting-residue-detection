{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "1BDMhzdOiT9L",
    "outputId": "8a15a315-e84c-4a3c-d06e-75cec8a369f1"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "     PDB_ID Residue Chain    ΔΔGobs Observed  PDBID_Chain\n",
      "0      1A4Y    W261     A       0.1        --      1A4Y.A\n",
      "1      1A4Y    W263     A       1.2         /      1A4Y.A\n",
      "2      1A4Y    S289     A         0        --      1A4Y.A\n",
      "3      1A4Y    W318     A       1.5         /      1A4Y.A\n",
      "4      1A4Y    K320     A      -0.3        --      1A4Y.A\n",
      "...     ...     ...   ...       ...       ...         ...\n",
      "2518   5E6P   Y1806     A  0.364633        --      5E6P.A\n",
      "2519   5E6P   A1832     A  0.118505        --      5E6P.A\n",
      "2520   5E6P   A1833     A  0.605117        --      5E6P.A\n",
      "2521   5E6P     I66     B  0.278581        --      5E6P.B\n",
      "2522   5E6P     R88     B      -0.6        --      5E6P.B\n",
      "\n",
      "[2523 rows x 6 columns]\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "# Load the CSV file into a DataFrame\n",
    "df = pd.read_excel('supplementary_table_S2_liu_et_al.xlsx')\n",
    "\n",
    "# Create the new column by concatenating PDB_ID and Chain with a period\n",
    "df['PDBID_Chain'] = df['PDB_ID'] + '.' + df['Chain']\n",
    "\n",
    "# Display the DataFrame to verify the new column\n",
    "print(df)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "rzMb5jWRauj9",
    "outputId": "9e556915-a026-4237-bf7f-23a4672e437f"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "    PDBID_Chain\n",
      "0        1A4Y.A\n",
      "1        1A4Y.B\n",
      "2        1AHW.C\n",
      "3        1BRS.A\n",
      "4        1BRS.D\n",
      "..          ...\n",
      "209      2NYY.C\n",
      "210      2NYY.D\n",
      "211      2NZ9.A\n",
      "212      2PTC.I\n",
      "213      2X0B.B\n",
      "\n",
      "[214 rows x 1 columns]\n"
     ]
    }
   ],
   "source": [
    "unique_values = df['PDBID_Chain'].unique()\n",
    "\n",
    "# Create a new DataFrame with these unique values\n",
    "unique_df1 = pd.DataFrame(unique_values, columns=['PDBID_Chain'])\n",
    "\n",
    "# Display the new DataFrame\n",
    "print(unique_df1)\n",
    "unique_df1.to_excel('output1.xlsx', index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "QRLiM3UJjjm1",
    "outputId": "4be8b1f7-b584-4910-febf-f4165ef10c01"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1A4Y.A 1A4Y.B 1AHW.C 1BRS.A 1BRS.D 1BXI.A 1CBW.D 1DAN.L 1DAN.T 1DAN.U 1DVF.A 1DVF.B 1DVF.D 3HHR.B 1DX5.M 1GC1.C 1JTG.A 1JTG.B 1NMB.H 1NMB.L 1VFB.B 3HFM.H 3HFM.L 3HFM.Y 3HHR.A 1CDL.A 1CDL.E 1DDM.A 1DDM.B 1DFJ.E 1DVA.H 1DVA.X 1DZI.A 1EBP.A 1EBP.C 1ES7.A 1FCC.C 1FOE.B 1GL4.A 1JAT.A 1JAT.B 1K4U.P 1LQB.D 1MQ8.B 1NFI.F 1UB4.C 2HHB.D 2NMB.A 2NMB.B 3SAK.A 1DX5.N 1FAK.T 1FE8.A 1G3I.A 1IHB.B 1JPP.B 1NUN.A 1A22.A 1A22.B 1AK4.D 1CBW.I 1CHO.I 1DAN.H 1DQJ.C 1DQJ.A 1DQJ.B 1DVF.C 1EAW.A 1EMV.A 1EMV.B 1F47.A 1FC2.C 1FFW.B 1GCQ.C 1H9D.B 1IAR.A 1JCK.B 1JRH.I 1JRH.L 1JRH.H 1KTZ.A 1KTZ.B 1LFD.A 1MAH.A 1PPF.I 1R0R.I 1REW.A 1REW.B 1REW.C 1S1Q.A 1TM1.I 1VFB.A 1VFB.C 1XD3.B 1Z7X.W 2FTL.I 2G2U.B 2I9B.E 2J0T.D 2J1K.C 2JEL.P 2O3B.B 2PCC.A 2QJA.C 2SIC.I 2VLJ.E 2WPT.A 2WPT.B 3BK3.C 3BN9.B 3BP8.A 3NPS.A 3SGB.I 4CPA.I 3BDY.H 3BDY.L 3BE1.H 3BE1.L 3BIW.A 3BN9.A 3BN9.H 3K2M.D 3MBW.A 3NGB.B 3NGB.C 3Q3J.B 3SE8.H 3SE8.L 3SE9.L 3W2D.A 4BFI.A 4BJ5.A 4BKL.A 4BPX.A 4BPX.B 4FZA.B 4HMY.C 4HSA.A 4IOF.A 4J2L.C 4JEU.A 4KRM.B 4L72.A 4LRX.A 4MYW.A 4NM8.B 4NZW.A 4NZW.B 4O27.B 4OFY.A 4OFY.D 4OZG.E 4OZG.F 4P23.C 4P23.D 4P4Q.A 4P4Q.B 4P5T.C 4P5T.D 4PWX.A 4PWX.C 4QTI.U 4RA0.C 4U6H.E 4UWQ.A 4X4M.E 4Y61.A 4Y61.B 4YEB.A 4YEB.B 4YFD.A 4YFD.B 5C6T.A 5C6T.H 5C6T.L 5CYK.A 5E6P.A 5E6P.B 5K39.B 1AXI.A 1BGS.A 1BGS.E 1BJ1.V 1BJ1.W 1CZ8.V 1CZ8.W 1DFJ.I 1F47.B 1ILP.C 1JTD.A 1JTD.B 1K8R.A 1MHP.H 1MHP.L 1MLC.A 1MLC.B 1MTN.D 1N8Z.A 1N8Z.B 1NVU.R 1S78.A 1U0S.Y 1U7F.B 1YCS.A 1YY9.C 1YY9.D 2KSO.A 2NYY.A 2NYY.C 2NYY.D 2NZ9.A 2PTC.I 2X0B.B\n"
     ]
    }
   ],
   "source": [
    "#all PDBID's\n",
    "df = pd.read_excel('output1.xlsx')\n",
    "pdbid_list = df['PDBID_Chain'].tolist()\n",
    "\n",
    "# Join the list into a single string separated by spaces\n",
    "pdbid_string = ' '.join(pdbid_list)\n",
    "\n",
    "# Print the resulting string\n",
    "print(pdbid_string)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "xvBPayW5konb",
    "outputId": "78358d95-83a3-4309-dddc-3d32f9c769ef"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Selected values (100): ['3HFM.L', '2NMB.B', '1DZI.A', '1JTG.A', '1JTG.B', '1CDL.A', '1BJ1.W', '1VFB.C', '5E6P.A', '1DDM.A', '1UB4.C', '2WPT.B', '4X4M.E', '1DX5.N', '1CHO.I', '4PWX.C', '1EAW.A', '1BGS.E', '2HHB.D', '1MTN.D', '4UWQ.A', '3HFM.Y', '4O27.B', '1N8Z.B', '1JAT.B', '3K2M.D', '1DVA.H', '4YFD.B', '1TM1.I', '2SIC.I', '4LRX.A', '1REW.A', '1DFJ.I', '1DDM.B', '1AXI.A', '1PPF.I', '4HMY.C', '1BGS.A', '3BN9.B', '1DVA.X', '2PCC.A', '3SAK.A', '3SE9.L', '1K8R.A', '4U6H.E', '5CYK.A', '2X0B.B', '3W2D.A', '1FAK.T', '4NM8.B', '1MHP.L', '1A4Y.A', '5C6T.A', '1F47.B', '1MLC.B', '1BRS.D', '1EBP.C', '3NPS.A', '1U0S.Y', '1Z7X.W', '1ES7.A', '2FTL.I', '4P23.C', '1DVF.B', '1DQJ.B', '2KSO.A', '1KTZ.A', '4FZA.B', '1A22.A', '3Q3J.B', '1JAT.A', '1MHP.H', '4Y61.A', '4J2L.C', '2G2U.B', '1KTZ.B', '3MBW.A', '1A4Y.B', '2NZ9.A', '4L72.A', '3HFM.H', '3BP8.A', '1U7F.B', '1REW.B', '1YY9.D', '4KRM.B', '4YEB.A', '3HHR.A', '3NGB.C', '3BE1.L', '1JRH.L', '1VFB.A', '3BE1.H', '1FCC.C', '4IOF.A', '1CBW.I', '5E6P.B', '1REW.C', '4OZG.F', '4JEU.A']\n",
      "Remaining values: ['1AHW.C', '1BRS.A', '1BXI.A', '1CBW.D', '1DAN.L', '1DAN.T', '1DAN.U', '1DVF.A', '1DVF.D', '3HHR.B', '1DX5.M', '1GC1.C', '1NMB.H', '1NMB.L', '1VFB.B', '1CDL.E', '1DFJ.E', '1EBP.A', '1FOE.B', '1GL4.A', '1K4U.P', '1LQB.D', '1MQ8.B', '1NFI.F', '2NMB.A', '1FE8.A', '1G3I.A', '1IHB.B', '1JPP.B', '1NUN.A', '1A22.B', '1AK4.D', '1DAN.H', '1DQJ.C', '1DQJ.A', '1DVF.C', '1EMV.A', '1EMV.B', '1F47.A', '1FC2.C', '1FFW.B', '1GCQ.C', '1H9D.B', '1IAR.A', '1JCK.B', '1JRH.I', '1JRH.H', '1LFD.A', '1MAH.A', '1R0R.I', '1S1Q.A', '1XD3.B', '2I9B.E', '2J0T.D', '2J1K.C', '2JEL.P', '2O3B.B', '2QJA.C', '2VLJ.E', '2WPT.A', '3BK3.C', '3SGB.I', '4CPA.I', '3BDY.H', '3BDY.L', '3BIW.A', '3BN9.A', '3BN9.H', '3NGB.B', '3SE8.H', '3SE8.L', '4BFI.A', '4BJ5.A', '4BKL.A', '4BPX.A', '4BPX.B', '4HSA.A', '4MYW.A', '4NZW.A', '4NZW.B', '4OFY.A', '4OFY.D', '4OZG.E', '4P23.D', '4P4Q.A', '4P4Q.B', '4P5T.C', '4P5T.D', '4PWX.A', '4QTI.U', '4RA0.C', '4Y61.B', '4YEB.B', '4YFD.A', '5C6T.H', '5C6T.L', '5K39.B', '1BJ1.V', '1CZ8.V', '1CZ8.W', '1ILP.C', '1JTD.A', '1JTD.B', '1MLC.A', '1N8Z.A', '1NVU.R', '1S78.A', '1YCS.A', '1YY9.C', '2NYY.A', '2NYY.C', '2NYY.D', '2PTC.I']\n",
      "Length of selected values: 100\n",
      "Length of remaining values: 113\n"
     ]
    }
   ],
   "source": [
    "import random\n",
    "assert len(df) >= 100, \"The DataFrame must contain at least 100 values.\"\n",
    "\n",
    "\n",
    "\n",
    "# Randomly select 100 unique values\n",
    "selected_values = random.sample(pdbid_list, 100)\n",
    "\n",
    "# Create the list of remaining values\n",
    "remaining_values = [value for value in pdbid_list if value not in selected_values]\n",
    "\n",
    "# Print the resulting lists\n",
    "print(\"Selected values (100):\", selected_values)\n",
    "print(\"Remaining values:\", remaining_values)\n",
    "\n",
    "# Optionally, verify the lengths of the lists\n",
    "print(\"Length of selected values:\", len(selected_values))\n",
    "print(\"Length of remaining values:\", len(remaining_values))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "8F6If6-Qk878",
    "outputId": "136867e5-e482-4387-c4c5-2c3cedb0ea59"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "3HFM.L 2NMB.B 1DZI.A 1JTG.A 1JTG.B 1CDL.A 1BJ1.W 1VFB.C 5E6P.A 1DDM.A 1UB4.C 2WPT.B 4X4M.E 1DX5.N 1CHO.I 4PWX.C 1EAW.A 1BGS.E 2HHB.D 1MTN.D 4UWQ.A 3HFM.Y 4O27.B 1N8Z.B 1JAT.B 3K2M.D 1DVA.H 4YFD.B 1TM1.I 2SIC.I 4LRX.A 1REW.A 1DFJ.I 1DDM.B 1AXI.A 1PPF.I 4HMY.C 1BGS.A 3BN9.B 1DVA.X 2PCC.A 3SAK.A 3SE9.L 1K8R.A 4U6H.E 5CYK.A 2X0B.B 3W2D.A 1FAK.T 4NM8.B 1MHP.L 1A4Y.A 5C6T.A 1F47.B 1MLC.B 1BRS.D 1EBP.C 3NPS.A 1U0S.Y 1Z7X.W 1ES7.A 2FTL.I 4P23.C 1DVF.B 1DQJ.B 2KSO.A 1KTZ.A 4FZA.B 1A22.A 3Q3J.B 1JAT.A 1MHP.H 4Y61.A 4J2L.C 2G2U.B 1KTZ.B 3MBW.A 1A4Y.B 2NZ9.A 4L72.A 3HFM.H 3BP8.A 1U7F.B 1REW.B 1YY9.D 4KRM.B 4YEB.A 3HHR.A 3NGB.C 3BE1.L 1JRH.L 1VFB.A 3BE1.H 1FCC.C 4IOF.A 1CBW.I 5E6P.B 1REW.C 4OZG.F 4JEU.A\n"
     ]
    }
   ],
   "source": [
    "#These are the training PDBIDs\n",
    "selected_values_string = ' '.join(selected_values)\n",
    "\n",
    "# Print the resulting string\n",
    "print(selected_values_string)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "QyP0BsPBlASS",
    "outputId": "0924c84f-d03f-46a3-e524-f515338f7081"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1AHW.C 1BRS.A 1BXI.A 1CBW.D 1DAN.L 1DAN.T 1DAN.U 1DVF.A 1DVF.D 3HHR.B 1DX5.M 1GC1.C 1NMB.H 1NMB.L 1VFB.B 1CDL.E 1DFJ.E 1EBP.A 1FOE.B 1GL4.A 1K4U.P 1LQB.D 1MQ8.B 1NFI.F 2NMB.A 1FE8.A 1G3I.A 1IHB.B 1JPP.B 1NUN.A 1A22.B 1AK4.D 1DAN.H 1DQJ.C 1DQJ.A 1DVF.C 1EMV.A 1EMV.B 1F47.A 1FC2.C 1FFW.B 1GCQ.C 1H9D.B 1IAR.A 1JCK.B 1JRH.I 1JRH.H 1LFD.A 1MAH.A 1R0R.I 1S1Q.A 1XD3.B 2I9B.E 2J0T.D 2J1K.C 2JEL.P 2O3B.B 2QJA.C 2VLJ.E 2WPT.A 3BK3.C 3SGB.I 4CPA.I 3BDY.H 3BDY.L 3BIW.A 3BN9.A 3BN9.H 3NGB.B 3SE8.H 3SE8.L 4BFI.A 4BJ5.A 4BKL.A 4BPX.A 4BPX.B 4HSA.A 4MYW.A 4NZW.A 4NZW.B 4OFY.A 4OFY.D 4OZG.E 4P23.D 4P4Q.A 4P4Q.B 4P5T.C 4P5T.D 4PWX.A 4QTI.U 4RA0.C 4Y61.B 4YEB.B 4YFD.A 5C6T.H 5C6T.L 5K39.B 1BJ1.V 1CZ8.V 1CZ8.W 1ILP.C 1JTD.A 1JTD.B 1MLC.A 1N8Z.A 1NVU.R 1S78.A 1YCS.A 1YY9.C 2NYY.A 2NYY.C 2NYY.D 2PTC.I\n"
     ]
    }
   ],
   "source": [
    "#These are the junction PDBIDs\n",
    "remaining_values_string = ' '.join(remaining_values)\n",
    "\n",
    "# Print the resulting string\n",
    "print(remaining_values_string)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "9RVYXBRn4PHa",
    "outputId": "b77ccde0-1ae2-444a-f6f6-e5de27f600ec"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "    PDBID\n",
      "0   DZI.A\n",
      "1   JTG.A\n",
      "2   JTG.B\n",
      "3   CDL.A\n",
      "4   WPT.B\n",
      "..    ...\n",
      "70  YEB.A\n",
      "71  C6T.A\n",
      "72  CYK.A\n",
      "73  E6P.A\n",
      "74  E6P.B\n",
      "\n",
      "[75 rows x 1 columns]\n"
     ]
    }
   ],
   "source": [
    "input_file = 'training_newest.csv'\n",
    "df1 = pd.read_csv(input_file)\n",
    "unique_values = df['PDBID'].unique()\n",
    "\n",
    "# Create a new DataFrame with these unique values\n",
    "unique_df1 = pd.DataFrame(unique_values, columns=['PDBID'])\n",
    "\n",
    "# Display the new DataFrame\n",
    "print(unique_df1)\n",
    "unique_df1.to_excel('PDBIDs_training.xlsx', index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "JYxiTs734lyE",
    "outputId": "dffbad5f-e47b-4185-a9f1-8ae8992eea88"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "    PDBID\n",
      "0   AHW.C\n",
      "1   BRS.A\n",
      "2   GC1.C\n",
      "3   CDL.E\n",
      "4   EBP.A\n",
      "..    ...\n",
      "76  PWX.A\n",
      "77  RA0.C\n",
      "78  Y61.B\n",
      "79  YEB.B\n",
      "80  K39.B\n",
      "\n",
      "[81 rows x 1 columns]\n"
     ]
    }
   ],
   "source": [
    "input_file = 'newjunction.csv'\n",
    "df1 = pd.read_csv(input_file)\n",
    "unique_values = df1['PDBID'].unique()\n",
    "\n",
    "# Create a new DataFrame with these unique values\n",
    "unique_df1 = pd.DataFrame(unique_values, columns=['PDBID'])\n",
    "\n",
    "# Display the new DataFrame\n",
    "print(unique_df1)\n",
    "unique_df1.to_excel('PDBID_junction.xlsx', index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "ROCboQOQ4vNv"
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
