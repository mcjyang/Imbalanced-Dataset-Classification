import pandas as pd
import numpy as np
import xgboost as xgb
from matplotlib import pylab as plt
from myfun import XGBFeatureImportances

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA

from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import *

# load data
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")

# remove constant columns (std = 0) # 34 constant
remove = []
for col in train.columns:
    if train[col].std() == 0:
        remove.append(col)
train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

# remove duplicated columns # 29 duplicated
remove = []
cols = train.columns
for i in range(len(cols)-1):
    v = train[cols[i]].values
    for j in range(i+1, len(cols)):
        if np.array_equal(v, train[cols[j]].values):
            remove.append(cols[j])
train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

# include sum and PCA
fs = train.columns[1:-1]
train.insert(1, 'amount_zeros', (train[fs] == 0).astype(int).sum(axis=1))
test.insert(1, 'amount_zeros', (test[fs] == 0).astype(int).sum(axis=1))

train['log_var38'] = np.log(train['var38'])
test['log_var38'] = np.log(test['var38'])
pca = PCA(n_components=2)
train_pca = pca.fit_transform(preprocessing.normalize(train[fs], axis=0))
test_pca = pca.transform(preprocessing.normalize(test[fs], axis=0))
train.insert(1, 'pca_1', train_pca[:, 0])
test.insert(1, 'pca_1', test_pca[:, 0])

test_id = test.ID
test = test.drop(["ID", 'var38'], axis=1)
X = train.drop(["TARGET", "ID", 'var38'], axis=1)
# test = test.drop(["ID"], axis=1)
# X = train.drop(["TARGET", "ID"], axis=1)
y = train.TARGET.values


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9527)
# print(x_train.shape, x_test.shape, test.shape)

# Random forest
model = RandomForestClassifier(n_estimators=150, max_depth=4, random_state=9527,verbose=1)
selector = model.fit(x_train, y_train)
importances = model.feature_importances_

# Lasso (Model based Ranking)
# a = -3
# model = linear_model.Lasso(alpha=10**a, random_state=9527)
# selector = model.fit(x_train, y_train)
# importances = model.coef_

# Ridge (Model based Ranking)
# a = -2
# model = linear_model.Ridge(alpha=10**a, random_state=9527)
# selector = model.fit(x_train, y_train)
# importances = model.coef_

# ExtraTree (Model based Ranking)
# model = ExtraTreesClassifier(random_state=9527)
# selector = model.fit(x_train, y_train)
# importances = model.feature_importances_

# DesicisonTree
# model = DecisionTreeClassifier( max_depth=4, random_state=9527)
# selector = model.fit(x_train, y_train)
# importances = model.feature_importances_

# Gradient descent booting Tree
# model = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=9527,verbose = 1)
# selector = model.fit(x_train, y_train)
# importances = model.feature_importances_

# XGB
model = xgb.XGBClassifier(n_estimators=150, nthread=-1, max_depth=4, seed=9527, subsample=0.9, colsample_bytree=0.9)
selector = model.fit(x_train, y_train, eval_metric="auc", verbose=False)
importances = model.feature_importances_


# Feature selection
fs = SelectFromModel(selector, prefit=True)
X_train = fs.transform(x_train)
X_test = fs.transform(x_test)
test = fs.transform(test)


# Show importance features w/o order
idx_fs = fs.get_support(indices=True)
# imp_fs = list(x_train[idx_fs].columns.values)
# for x in imp_fs:
#     print x

# Plot feature importances
# std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
# plt.figure()
# plt.title("Feature Importances")
# plt.barh(range(30), importances[indices][:30][::-1], color="b", align="center")
# plt.yticks(range(30), x_train.columns[indices[:30]].values[::-1])
# plt.ylim([-1, 30])
# plt.savefig('lasso_o_fs.png')
# plt.show()
print x_train.columns[indices[:30]].values[::-1]

# Train Model (classifier from xgboost)
# model = xgb.XGBClassifier(n_estimators=150, nthread=-1, max_depth=4, seed=9527)
# model.fit(X_train, y_train, eval_metric="auc", verbose=False, eval_set=[(X_test, y_test)])
# print("Roc AUC: ", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1], average='macro'))

# model = linear_model.Ridge(alpha=10**a, random_state=9527)
# model.fit(X_train, y_train)
# res = model.predict(X_test)
# print("Roc AUC: ", roc_auc_score(y_test, res))


# SVM
# model = SVC(verbose=True, max_iter=5000, C=1000, random_state=9527)
# model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), algorithm='SAMME', n_estimators=150, random_state=9527)
# model.fit(X_train, y_train)
# res = model.predict(X_test)
# print res
# print("Roc AUC: ", roc_auc_score(y_test, res))

# Submission
# probs = model.predict_proba(test)
# submission = pd.DataFrame({"ID": test_id, "TARGET": probs[:, 1]})
# submission.to_csv("submission.csv", index=False)

print 'end'