import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.ensemble import *
from sklearn.tree import *
from sklearn.svm import *
from scipy import stats

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

fs = train.columns[1:-1]
train.insert(1, 'sumZeros', (train[fs] == 0).astype(int).sum(axis=1))
test.insert(1, 'sumZeros', (test[fs] == 0).astype(int).sum(axis=1))

train['log_var38'] = np.log(train['var38'])
test['log_var38'] = np.log(test['var38'])

pca = PCA(n_components=2)
train_pca = pca.fit_transform(preprocessing.normalize(train[fs], axis=0))
test_pca = pca.transform(preprocessing.normalize(test[fs], axis=0))
train.insert(1, 'pca_1',train_pca[:,0])
test.insert(1, 'pca_1',test_pca[:,0])

fs_columns = ['var15', 'log_var38', 'pca_1',
              'sumZeros', 'saldo_medio_var5_hace3', 'saldo_medio_var5_ult3',
              'num_var45_hace2', 'num_var45_hace3',  'saldo_var30']
x_train = train[fs_columns]
y_train = train['TARGET']
x_train = train.drop(['TARGET', 'ID', 'var38'], axis=1)
# y_train = train['TARGET']
# selected_test = test[fs_columns]

X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.20, random_state=9527)

# XGB
model = xgb.XGBClassifier(n_estimators=100, nthread=-1, max_depth=4, seed=9527, subsample=0.9, colsample_bytree=0.9)
model.fit(X_train, y_train, eval_metric="auc", verbose=False, eval_set=[(X_test, y_test)])
# print("Roc AUC: ", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1],average='macro'))

model = AdaBoostClassifier(n_estimators=110, random_state=9527)
model.fit(X_train, y_train)
res = model.predict(X_test)
print("Roc AUC: ", roc_auc_score(y_test, res))

# Submission
# probs = m2_xgb.predict_proba(selected_test)
# submission = pd.DataFrame({"ID": test['ID'], "TARGET": probs[:, 1]})
# submission.to_csv("submission.csv", index='False')

