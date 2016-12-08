import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score
from sklearn import svm

from scipy import stats

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
train.insert(1, 'sumZeros', (train[fs] == 0).astype(int).sum(axis=1))
test.insert(1, 'sumZeros', (test[fs] == 0).astype(int).sum(axis=1))

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
y = train.TARGET.values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9527)
train_size = x_train.shape[0]

model = ExtraTreesClassifier(random_state=9527)
selector = model.fit(x_train, y_train)

fs = SelectFromModel(selector, prefit=True)
X_train = fs.transform(x_train)
X_test = fs.transform(x_test)
test = fs.transform(test)

print 'start one classifier'
outliers_fraction = 4.0/100
n_outliers = train_size * outliers_fraction

# One-Classifier
model = svm.OneClassSVM(nu=0.95 * outliers_fraction+0.05, kernel="rbf", gamma=0.1, verbose=True)
model.fit(preprocessing.scale(X_train, axis=0))
X_dec = model.decision_function(preprocessing.scale(X_test, axis=0))
threshold = stats.scoreatpercentile(X_dec, 100 * outliers_fraction)
test = [0 if x > threshold else 1 for x in X_dec]
print len([x for x in test if x == 1])

# calculate the auc score
print("Roc AUC: ", roc_auc_score(y_test, test, average='macro'))
