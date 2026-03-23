# %%
from joblib import load
from sklearn.metrics import precision_recall_curve, auc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
import lightgbm as lgb
from joblib import dump
import itertools
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import RandomOverSampler

rice_df = pd.read_excel('rice_w2vec_3mer_64_dataset.xlsx')
rice_df = rice_df.drop(['circName','tissue','chr','start','end','strand','start_anno', 'circID',	'gene', 'isoform', 'width', 'detection_score', 'stress_detection_score', 'end_anno', 'antisense', 'algorithm', 'seq', 'exonSeq', 'predAA', 'miRNA','superCircRNARegion'], axis=1)

maize_df = pd.read_excel('maize_w2vec_3mer_64_dataset.xlsx')
maize_df = maize_df.drop(['circName','tissue','chr','start','end','strand','start_anno', 'circID',	'gene', 'isoform', 'width', 'detection_score', 'stress_detection_score', 'end_anno', 'antisense', 'algorithm', 'seq', 'exonSeq', 'predAA', 'miRNA','superCircRNARegion'], axis=1)

df = rice_df._append(maize_df, ignore_index = True)

df_drought = df.query('stress == "-" or stress == "drought"').replace('-', 0).replace('drought', 1)

# Define X and y (target) variables
X = df_drought.drop(['stress'], axis=1)
y = df_drought['stress']

ros = RandomUnderSampler(sampling_strategy=0.55)

X_res, y_res = ros.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.20, random_state=106)

rus = RandomUnderSampler(sampling_strategy=1)
X_test, y_test = rus.fit_resample(X_test, y_test)

ros = RandomOverSampler(sampling_strategy=1)
X_train, y_train = ros.fit_resample(X_train, y_train)


# 1. Load the pre-trained models
# Ensure these files are in your current working directory
model_rf = load('rf_model_drought_3mer.joblib')
model_lgb = load('lgb_model_drought_3mer.joblib')

# 2. Get prediction probabilities for the positive class (Stress)
# We use [:, 1] to get the probability of the sequence being 'Stress'
probs_rf = model_rf.predict_proba(X_test)[:, 1]
probs_lgb = model_lgb.predict_proba(X_test)[:, 1]

# 3. Calculate AUPR for Random Forest
precision_rf, recall_rf, _ = precision_recall_curve(y_test, probs_rf)
aupr_rf = auc(recall_rf, precision_rf)

# 4. Calculate AUPR for LightGBM
precision_lgb, recall_lgb, _ = precision_recall_curve(y_test, probs_lgb)
aupr_lgb = auc(recall_lgb, precision_lgb)

# 5. Clear Numerical Output
print("-" * 40)
print("PERFORMANCE METRICS: AUPR (Area Under PR)")
print("-" * 40)
print(f"Random Forest (RF):   {aupr_rf:.4f}")
print(f"LightGBM (LGBM):      {aupr_lgb:.4f}")
print("-" * 40)

# Optional: Calculate the improvement
diff = ((aupr_lgb - aupr_rf) / aupr_rf) * 100
print(f"Difference: LightGBM is {diff:+.2f}% relative to RF")
# %%

# Results

#Random Forest (RF):   0.8151
#LightGBM (LGBM):      0.8039
