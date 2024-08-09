{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "b92e2f11",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "7f6667fa",
   "metadata": {},
   "outputs": [],
   "source": [
    "# To find out junctions\n",
    "def identify_junction_regions(data_frame):\n",
    "    # Create a new column 'Junction_Region'\n",
    "    data_frame['Junction_Region'] = data_frame['Secondary_Structure']\n",
    "\n",
    "    # Identify and mark the junction regions\n",
    "    for i in range(1, len(data_frame)):\n",
    "        current_structure = data_frame.at[i, 'Secondary_Structure']\n",
    "        previous_structure = data_frame.at[i - 1, 'Secondary_Structure']\n",
    "\n",
    "        if current_structure != previous_structure:\n",
    "            data_frame.at[i, 'Junction_Region'] = 'J'\n",
    "            data_frame.at[i - 1, 'Junction_Region'] = 'J'\n",
    "\n",
    "    return data_frame\n",
    "\n",
    "# Replace 'your_file.xlsx' with the actual file name\n",
    "file_path = 'training.csv'\n",
    "\n",
    "# Read the Excel file into a DataFrame\n",
    "df = pd.read_csv(file_path)\n",
    "\n",
    "# Apply the function to identify junction regions\n",
    "df = identify_junction_regions(df)\n",
    "\n",
    "# Save the modified DataFrame to a new Excel file\n",
    "df.to_csv('newtraining.csv', index=False)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "1b33fa21",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "New CSV file saved as training.csv\n"
     ]
    }
   ],
   "source": [
    "# To assign secondary structure propensity values\n",
    "input_filename = 'FINAL_training.csv'\n",
    "output_filename = 'training.csv'\n",
    "\n",
    "# Read the CSV file into a DataFrame\n",
    "df = pd.read_csv(input_filename)\n",
    "\n",
    "# Function to map values based on conditions\n",
    "def map_secondary_structure(value):\n",
    "    if value == 'C':\n",
    "        return 0.4333\n",
    "    elif value == 'H':\n",
    "        return 0.1307\n",
    "    elif value == 'J':\n",
    "        return 6.2753\n",
    "    elif value == 'E':\n",
    "        return 0.17134\n",
    "    else:\n",
    "        return 'X'\n",
    "\n",
    "# Apply the mapping function to create a new column 'New_Secondary_Structure'\n",
    "df['Secondary_Enum'] = df['Junction_Region'].apply(map_secondary_structure)\n",
    "\n",
    "# Save the modified DataFrame to a new CSV file\n",
    "df.to_csv(output_filename, index=False)\n",
    "\n",
    "print(f\"New CSV file saved as {output_filename}\")"
   ]
  }
 ],
 "metadata": {
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
 "nbformat_minor": 5
}
