import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from joblib import dump

rice_df = pd.read_excel('rice_word2vec_3mer_dataset.xlsx')
maize_df = pd.read_excel('maize_word2vec_3mer_dataset.xlsx')

df = rice_df._append(maize_df, ignore_index = True)

df_multiclasse = df.query('stress == "-" or stress == "drought" or  stress == "cold"').replace('drought', 1).replace('cold', 2).replace('-', 0)

X = df_multiclasse.drop(['circName','stress','tissue','chr','start','end','strand','start_anno'], axis=1)
y = df_multiclasse['stress']

ros = RandomOverSampler(sampling_strategy = {0: 70996, 1: 12030, 2: 10000})

X_res, y_res = ros.fit_resample(X, y)

rus = RandomUnderSampler(sampling_strategy = {0: 12030, 1: 12030, 2: 10000})

X_res, y_res = rus.fit_resample(X_res, y_res)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.20)

modelrf = RandomForestClassifier()
modelrf.fit(X_train, y_train)

y_pred = modelrf.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)

# Saving the model

dump(modelrf, 'rf_model.joblib')
print('Multiclass Model created.')