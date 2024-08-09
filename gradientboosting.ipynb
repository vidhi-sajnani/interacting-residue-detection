{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting imbalanced-learn\n",
      "  Obtaining dependency information for imbalanced-learn from https://files.pythonhosted.org/packages/5a/fa/267de06c95210580f4b82b45cec1ce1e9ce1f21a01a684367db89e7da70d/imbalanced_learn-0.12.3-py3-none-any.whl.metadata\n",
      "  Downloading imbalanced_learn-0.12.3-py3-none-any.whl.metadata (8.3 kB)\n",
      "Requirement already satisfied: numpy>=1.17.3 in c:\\users\\vidhi sajnani\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from imbalanced-learn) (1.25.2)\n",
      "Requirement already satisfied: scipy>=1.5.0 in c:\\users\\vidhi sajnani\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from imbalanced-learn) (1.11.2)\n",
      "Requirement already satisfied: scikit-learn>=1.0.2 in c:\\users\\vidhi sajnani\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from imbalanced-learn) (1.3.0)\n",
      "Requirement already satisfied: joblib>=1.1.1 in c:\\users\\vidhi sajnani\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from imbalanced-learn) (1.3.2)\n",
      "Requirement already satisfied: threadpoolctl>=2.0.0 in c:\\users\\vidhi sajnani\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from imbalanced-learn) (3.2.0)\n",
      "Downloading imbalanced_learn-0.12.3-py3-none-any.whl (258 kB)\n",
      "   -------------------------------------- 258.3/258.3 kB 992.3 kB/s eta 0:00:00\n",
      "Installing collected packages: imbalanced-learn\n",
      "Successfully installed imbalanced-learn-0.12.3\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "[notice] A new release of pip is available: 23.2.1 -> 24.0\n",
      "[notice] To update, run: python.exe -m pip install --upgrade pip\n"
     ]
    }
   ],
   "source": [
    "pip install -U imbalanced-learn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "id": "9yNeNtreNWAh"
   },
   "outputs": [],
   "source": [
    "# Import models and utility functions\n",
    "from sklearn.ensemble import GradientBoostingClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "# Importing imblearn, scikit-learn library\n",
    "from imblearn.over_sampling import SMOTE\n",
    "from sklearn.datasets import make_classification\n",
    "from imblearn.over_sampling import RandomOverSampler\n",
    "from imblearn.under_sampling import RandomUnderSampler\n",
    "from collections import Counter\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 443
    },
    "id": "o8cU1pqAOD-k",
    "outputId": "e679d8f4-a9c6-44b6-d863-793629242792"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Amino_Acid</th>\n",
       "      <th>Arginine_Propensity</th>\n",
       "      <th>Tryptophan_Propensity</th>\n",
       "      <th>Tyrosine_Propensity</th>\n",
       "      <th>Valine_Propensity</th>\n",
       "      <th>Serine_Propensity</th>\n",
       "      <th>Methionine_Propensity</th>\n",
       "      <th>Threonine_Propensity</th>\n",
       "      <th>Leucine_Propensity</th>\n",
       "      <th>RelSASA</th>\n",
       "      <th>Secondary_Enum</th>\n",
       "      <th>Label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>0.738996</td>\n",
       "      <td>0.43330</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>10</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>0.417377</td>\n",
       "      <td>0.43330</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>8</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>0.110747</td>\n",
       "      <td>6.27530</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-0.428571</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.142857</td>\n",
       "      <td>0.094342</td>\n",
       "      <td>6.27530</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>18</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-0.571429</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.142857</td>\n",
       "      <td>0.016116</td>\n",
       "      <td>0.17134</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17990</th>\n",
       "      <td>6</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.285714</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.142857</td>\n",
       "      <td>-0.285714</td>\n",
       "      <td>0.424894</td>\n",
       "      <td>0.43330</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17991</th>\n",
       "      <td>16</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.428571</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.285714</td>\n",
       "      <td>0.603341</td>\n",
       "      <td>0.43330</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17992</th>\n",
       "      <td>16</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.500000</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.166667</td>\n",
       "      <td>0.689852</td>\n",
       "      <td>0.43330</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17993</th>\n",
       "      <td>13</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.600000</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>0.742278</td>\n",
       "      <td>0.43330</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17994</th>\n",
       "      <td>16</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.750000</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>0.787751</td>\n",
       "      <td>0.43330</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>17995 rows × 12 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       Amino_Acid  Arginine_Propensity  Tryptophan_Propensity  \\\n",
       "0               1                  0.0                    0.0   \n",
       "1              10                  0.0                    0.0   \n",
       "2               8                  0.0                    0.0   \n",
       "3               3                  0.0                    0.0   \n",
       "4              18                  0.0                    0.0   \n",
       "...           ...                  ...                    ...   \n",
       "17990           6                  0.0                    0.0   \n",
       "17991          16                  0.0                    0.0   \n",
       "17992          16                  0.0                    0.0   \n",
       "17993          13                  0.0                    0.0   \n",
       "17994          16                  0.0                    0.0   \n",
       "\n",
       "       Tyrosine_Propensity  Valine_Propensity  Serine_Propensity  \\\n",
       "0                      0.0          -0.000000          -0.000000   \n",
       "1                      0.0          -0.000000          -0.000000   \n",
       "2                      0.0          -0.000000          -0.000000   \n",
       "3                      0.0          -0.428571          -0.000000   \n",
       "4                      0.0          -0.571429          -0.000000   \n",
       "...                    ...                ...                ...   \n",
       "17990                  0.0          -0.000000          -0.285714   \n",
       "17991                  0.0          -0.000000          -0.428571   \n",
       "17992                  0.0          -0.000000          -0.500000   \n",
       "17993                  0.0          -0.000000          -0.600000   \n",
       "17994                  0.0          -0.000000          -0.750000   \n",
       "\n",
       "       Methionine_Propensity  Threonine_Propensity  Leucine_Propensity  \\\n",
       "0                       -0.0             -0.000000           -0.000000   \n",
       "1                       -0.0             -0.000000           -0.000000   \n",
       "2                       -0.0             -0.000000           -0.000000   \n",
       "3                       -0.0             -0.000000           -0.142857   \n",
       "4                       -0.0             -0.000000           -0.142857   \n",
       "...                      ...                   ...                 ...   \n",
       "17990                   -0.0             -0.142857           -0.285714   \n",
       "17991                   -0.0             -0.000000           -0.285714   \n",
       "17992                   -0.0             -0.000000           -0.166667   \n",
       "17993                   -0.0             -0.000000           -0.000000   \n",
       "17994                   -0.0             -0.000000           -0.000000   \n",
       "\n",
       "        RelSASA  Secondary_Enum  Label  \n",
       "0      0.738996         0.43330    0.0  \n",
       "1      0.417377         0.43330    0.0  \n",
       "2      0.110747         6.27530    0.0  \n",
       "3      0.094342         6.27530    0.0  \n",
       "4      0.016116         0.17134    0.0  \n",
       "...         ...             ...    ...  \n",
       "17990  0.424894         0.43330    0.0  \n",
       "17991  0.603341         0.43330    1.0  \n",
       "17992  0.689852         0.43330    1.0  \n",
       "17993  0.742278         0.43330    1.0  \n",
       "17994  0.787751         0.43330    1.0  \n",
       "\n",
       "[17995 rows x 12 columns]"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv(\"final_training.csv\")\n",
    "df['Label'] = df['Label'].replace(np.nan, 0)\n",
    "df=df.drop(['Secondary_Structure',\"Residue_Index\", \"PDBID\",\"Junction_Region\",\"Chain\"], axis=1)\n",
    "df[\"Valine_Propensity\"]=df[\"Valine_Propensity\"].apply(lambda x:x*-1)\n",
    "df[\"Serine_Propensity\"]=df[\"Serine_Propensity\"].apply(lambda x:x*-1)\n",
    "df[\"Methionine_Propensity\"]=df[\"Methionine_Propensity\"].apply(lambda x:x*-1)\n",
    "df[\"Threonine_Propensity\"]=df[\"Threonine_Propensity\"].apply(lambda x:x*-1)\n",
    "df[\"Leucine_Propensity\"]=df[\"Leucine_Propensity\"].apply(lambda x:x*-1)\n",
    "aa_to_number = {\n",
    "    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,\n",
    "    'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,\n",
    "    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,\n",
    "    'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20\n",
    "}\n",
    "# label=\"PDBID\"\n",
    "# classes= df[label].unique().tolist()\n",
    "# print(f\"Label classes: {classes}\")\n",
    "# df[label]= df[label].map(classes.index)\n",
    "\n",
    "# Convert Amino_Acid column to numbers\n",
    "df['Amino_Acid'] = df['Amino_Acid'].map(aa_to_number)\n",
    "df\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "id": "1X6wmv2DOPQl"
   },
   "outputs": [],
   "source": [
    "X= df.drop('Label', axis=1)\n",
    "y=df[\"Label\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "id": "uxL2fwN4Pke6"
   },
   "outputs": [],
   "source": [
    "from imblearn.over_sampling import SMOTE \n",
    "sm = SMOTE(random_state = 2) \n",
    "\n",
    "# Fit predictor (x variable)\n",
    "# and target (y variable) using fit_resample()\n",
    "X_resampled, y_resampled = sm.fit_resample(X, y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "yz3kBJg652Qr",
    "outputId": "5abbe3d0-6ef4-4666-c92c-b4ef973efbf7"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Gradient Boosting Classifier accuracy is : 0.90\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "         0.0       0.90      1.00      0.94      3222\n",
      "         1.0       1.00      0.00      0.01       377\n",
      "\n",
      "    accuracy                           0.90      3599\n",
      "   macro avg       0.95      0.50      0.48      3599\n",
      "weighted avg       0.91      0.90      0.85      3599\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# WITHOUT SMOTE\n",
    "# Splitting dataset\n",
    "train_Xo, test_Xo, train_yo, test_yo = train_test_split(X, y, test_size = 0.20)\n",
    "\n",
    "# Instantiate Gradient Boosting Regressor\n",
    "gbc = GradientBoostingClassifier(n_estimators=300,\n",
    "                                 learning_rate=0.01,\n",
    "                                 random_state=100,\n",
    "                                 max_features=13 )\n",
    "# Fit to training set\n",
    "gbc.fit(train_Xo, train_yo)\n",
    "\n",
    "# Predict on test set\n",
    "pred_yo = gbc.predict(test_Xo)\n",
    "\n",
    "# accuracy\n",
    "acc = accuracy_score(test_yo, pred_yo)\n",
    "print(\"Gradient Boosting Classifier accuracy is : {:.2f}\".format(acc))\n",
    "from sklearn.metrics import classification_report\n",
    "print(classification_report(test_yo, pred_yo))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "7_yyyNdWOCWk",
    "outputId": "55bfb54b-0052-4d33-82a3-16d31b48321e"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Gradient Boosting Classifier accuracy is : 0.87\n"
     ]
    }
   ],
   "source": [
    "# WITH SMOTE\n",
    "# Splitting dataset\n",
    "train_X, test_X, train_y, test_y = train_test_split(X_resampled, y_resampled, test_size = 0.20)\n",
    "\n",
    "# Instantiate Gradient Boosting Regressor\n",
    "gbc = GradientBoostingClassifier(n_estimators=300,\n",
    "                                 learning_rate=0.01,\n",
    "                                 random_state=100,\n",
    "                                 max_features=13 )\n",
    "# Fit to training set\n",
    "gbc.fit(train_X, train_y)\n",
    "\n",
    "# Predict on test set\n",
    "pred_y = gbc.predict(test_X)\n",
    "\n",
    "# accuracy\n",
    "acc = accuracy_score(test_y, pred_y)\n",
    "print(\"Gradient Boosting Classifier accuracy is : {:.2f}\".format(acc))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "8rG9ZqmTOZg1",
    "outputId": "b8eb6c52-5de2-492e-c655-b0dd95e8a164"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "         0.0       0.80      0.98      0.89      3203\n",
      "         1.0       0.98      0.77      0.86      3255\n",
      "\n",
      "    accuracy                           0.87      6458\n",
      "   macro avg       0.89      0.87      0.87      6458\n",
      "weighted avg       0.89      0.87      0.87      6458\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "print(classification_report(test_y, pred_y))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "n8MDbwpZSbVF",
    "outputId": "028d6211-2adb-43be-af66-469d7a5b068b"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9271523178807947"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import fbeta_score\n",
    "fbeta_score(test_y, pred_y, average='binary', beta=0.5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "THOTQAuqadJN",
    "outputId": "35ae7240-7a9c-476d-861c-2077df662796"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[3149,   54],\n",
       "       [ 763, 2492]], dtype=int64)"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "confusion_matrix(test_y, pred_y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 461
    },
    "id": "jsmc5o4cO8LA",
    "outputId": "af60acbd-2fab-459e-ba1a-295f9daba49f"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Amino_Acid</th>\n",
       "      <th>Arginine_Propensity</th>\n",
       "      <th>Tryptophan_Propensity</th>\n",
       "      <th>Tyrosine_Propensity</th>\n",
       "      <th>Valine_Propensity</th>\n",
       "      <th>Serine_Propensity</th>\n",
       "      <th>Methionine_Propensity</th>\n",
       "      <th>Threonine_Propensity</th>\n",
       "      <th>Leucine_Propensity</th>\n",
       "      <th>RelSASA</th>\n",
       "      <th>Secondary_Enum</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>11</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>0.783568</td>\n",
       "      <td>0.550000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>0.742773</td>\n",
       "      <td>0.550000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>15</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>0.592849</td>\n",
       "      <td>0.276254</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8</td>\n",
       "      <td>0.142857</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.142857</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.142857</td>\n",
       "      <td>0.516733</td>\n",
       "      <td>0.276254</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>9</td>\n",
       "      <td>0.285714</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.142857</td>\n",
       "      <td>0.554655</td>\n",
       "      <td>0.276254</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3317</th>\n",
       "      <td>16</td>\n",
       "      <td>0.142857</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.142857</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.142857</td>\n",
       "      <td>0.435100</td>\n",
       "      <td>0.276250</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3318</th>\n",
       "      <td>5</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.142857</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.142857</td>\n",
       "      <td>0.731220</td>\n",
       "      <td>0.276250</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3319</th>\n",
       "      <td>14</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>0.695062</td>\n",
       "      <td>0.550000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3320</th>\n",
       "      <td>10</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>0.552793</td>\n",
       "      <td>0.550000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3321</th>\n",
       "      <td>8</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>-0.0</td>\n",
       "      <td>-0.000000</td>\n",
       "      <td>0.837853</td>\n",
       "      <td>0.460070</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>3322 rows × 11 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      Amino_Acid  Arginine_Propensity  Tryptophan_Propensity  \\\n",
       "0             11             0.000000                    0.0   \n",
       "1              4             0.000000                    0.0   \n",
       "2             15             0.000000                    0.0   \n",
       "3              8             0.142857                    0.0   \n",
       "4              9             0.285714                    0.0   \n",
       "...          ...                  ...                    ...   \n",
       "3317          16             0.142857                    0.0   \n",
       "3318           5             0.000000                    0.0   \n",
       "3319          14             0.000000                    0.0   \n",
       "3320          10             0.000000                    0.0   \n",
       "3321           8             0.000000                    0.0   \n",
       "\n",
       "      Tyrosine_Propensity  Valine_Propensity  Serine_Propensity  \\\n",
       "0                     0.0               -0.0          -0.000000   \n",
       "1                     0.0               -0.0          -0.000000   \n",
       "2                     0.0               -0.0          -0.000000   \n",
       "3                     0.0               -0.0          -0.000000   \n",
       "4                     0.0               -0.0          -0.000000   \n",
       "...                   ...                ...                ...   \n",
       "3317                  0.0               -0.0          -0.142857   \n",
       "3318                  0.0               -0.0          -0.142857   \n",
       "3319                  0.0               -0.0          -0.000000   \n",
       "3320                  0.0               -0.0          -0.000000   \n",
       "3321                  0.0               -0.0          -0.000000   \n",
       "\n",
       "      Methionine_Propensity  Threonine_Propensity  Leucine_Propensity  \\\n",
       "0                 -0.000000                  -0.0           -0.000000   \n",
       "1                 -0.000000                  -0.0           -0.000000   \n",
       "2                 -0.000000                  -0.0           -0.000000   \n",
       "3                 -0.142857                  -0.0           -0.142857   \n",
       "4                 -0.000000                  -0.0           -0.142857   \n",
       "...                     ...                   ...                 ...   \n",
       "3317              -0.000000                  -0.0           -0.142857   \n",
       "3318              -0.000000                  -0.0           -0.142857   \n",
       "3319              -0.000000                  -0.0           -0.000000   \n",
       "3320              -0.000000                  -0.0           -0.000000   \n",
       "3321              -0.000000                  -0.0           -0.000000   \n",
       "\n",
       "       RelSASA  Secondary_Enum  \n",
       "0     0.783568        0.550000  \n",
       "1     0.742773        0.550000  \n",
       "2     0.592849        0.276254  \n",
       "3     0.516733        0.276254  \n",
       "4     0.554655        0.276254  \n",
       "...        ...             ...  \n",
       "3317  0.435100        0.276250  \n",
       "3318  0.731220        0.276250  \n",
       "3319  0.695062        0.550000  \n",
       "3320  0.552793        0.550000  \n",
       "3321  0.837853        0.460070  \n",
       "\n",
       "[3322 rows x 11 columns]"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df1 = pd.read_excel(\"viral_test_dataset.xlsx\")\n",
    "df1 = df1.drop([\"Residue_Index\", \"PDBID\"], axis=1)\n",
    "df1[\"Valine_Propensity\"]=df1[\"Valine_Propensity\"].apply(lambda x:x*-1)\n",
    "df1[\"Serine_Propensity\"]=df1[\"Serine_Propensity\"].apply(lambda x:x*-1)\n",
    "df1[\"Methionine_Propensity\"]=df1[\"Methionine_Propensity\"].apply(lambda x:x*-1)\n",
    "df1[\"Threonine_Propensity\"]=df1[\"Threonine_Propensity\"].apply(lambda x:x*-1)\n",
    "df1[\"Leucine_Propensity\"]=df1[\"Leucine_Propensity\"].apply(lambda x:x*-1)\n",
    "# label=\"PDBID\"\n",
    "# classes= df1[label].unique().tolist()\n",
    "# print(f\"Label classes: {classes}\")\n",
    "# df1[label]= df1[label].map(classes.index)\n",
    "aa_to_number = {\n",
    "    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,\n",
    "    'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,\n",
    "    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,\n",
    "    'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20\n",
    "}\n",
    "\n",
    "# Convert Amino_Acid column to numbers\n",
    "df1['Amino_Acid'] = df1['Amino_Acid'].map(aa_to_number)\n",
    "df1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "Cj4QhCVTRkUT",
    "outputId": "bfad1161-2e33-4d76-cd02-e0e66a22e7cf"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predicted Labels:\n",
      "[0. 0. 0. ... 0. 0. 0.]\n"
     ]
    }
   ],
   "source": [
    "# Assuming 'new_data' is your new dataset\n",
    "predictions = gbc.predict(df1)\n",
    "\n",
    "# If you want to print the predicted labels\n",
    "print(\"Predicted Labels:\")\n",
    "print(predictions)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "gFZwjX9lRq-q",
    "outputId": "42d2ac59-9ed0-4dba-8cbd-7d5db7ae1f8f"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predictions saved to gb-preds.csv\n"
     ]
    }
   ],
   "source": [
    "csv_file_path = \"gb-preds.csv\"\n",
    "\n",
    "# Create a DataFrame with the predicted labels\n",
    "result_df = pd.DataFrame({'Predicted_Labels': predictions.flatten()})\n",
    "\n",
    "# Save the DataFrame to a CSV file\n",
    "result_df.to_csv(csv_file_path, index=False)\n",
    "\n",
    "# Print a message indicating the successful save\n",
    "print(f\"Predictions saved to {csv_file_path}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {
    "id": "aAf0a17aR05N"
   },
   "outputs": [],
   "source": [
    "df2 = pd.read_excel(\"external-val.xlsx\")\n",
    "df2['LUBNA'] = df2['LUBNA'].replace(np.nan, 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {
    "id": "9NpjobniOP-y"
   },
   "outputs": [],
   "source": [
    "test=df2['LUBNA']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "gZ6lED0sOXmS",
    "outputId": "1712363c-70f0-4afd-938e-257b1a37bc06"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "         0.0       0.58      1.00      0.73      1927\n",
      "         1.0       1.00      0.00      0.00      1395\n",
      "\n",
      "    accuracy                           0.58      3322\n",
      "   macro avg       0.79      0.50      0.37      3322\n",
      "weighted avg       0.76      0.58      0.43      3322\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(test, predictions))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "CohWlr1UObG1"
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
