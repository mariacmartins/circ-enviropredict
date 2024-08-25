import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

rice_df = pd.read_excel('rice_word2vec_3mer_dataset.xlsx')
maize_df = pd.read_excel('maize_word2vec_3mer_dataset.xlsx')

df = rice_df._append(maize_df, ignore_index = True)

df_drought = df.query('stress == "-" or stress == "drought"').replace('-', 0).replace('drought', 1)

# Define X and y (target) variables
X = df_drought.drop(['circName','stress','tissue','chr','start','end','strand','start_anno'], axis=1)
y = df_drought['stress']

ros = RandomUnderSampler(sampling_strategy=1)

X_res, y_res = ros.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.20)

modelrf = RandomForestClassifier()
modelrf.fit(X_train, y_train)

y_pred = modelrf.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)

# Saving the model

dump(modelrf, 'rf_model_drought.joblib')
print('Drought Model created.')