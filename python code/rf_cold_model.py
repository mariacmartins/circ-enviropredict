import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

rice_df = pd.read_excel('data/word2vec/rice_w2vec_3mer_dataset.xlsx')

df_cold = rice_df.query('stress == "-" or stress == "cold"').replace('-', 0).replace('cold', 1)

# Define X and y (target) variables
X = df_cold.drop(['circName','stress','tissue','chr','start','end','strand','start_anno', 'circID',	'gene', 'isoform', 'width', 'detection_score', 'stress_detection_score', 'end_anno', 'antisense', 'algorithm', 'seq', 'exonSeq', 'predAA', 'miRNA','superCircRNARegion'], axis=1)
y = df_cold['stress']

ros = RandomUnderSampler(sampling_strategy=0.7)

X_res, y_res = ros.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.20, random_state=320)

modelrf = RandomForestClassifier(random_state=320)
modelrf.fit(X_train, y_train)

y_pred = modelrf.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)

# Saving the model

dump(modelrf, 'rf_model_cold_3mer.joblib')
print('Cold Model created.')