import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import svm
import os

dining_path = "../data/dining/"
running_path = "../data/running/"
dining = []
running = []

for f in os.listdir(dining_path):
    for line in open(dining_path + f, 'r'):
        line = line.strip()
        list_ = line.split(',')
        del list_[0]
        list_ = map(lambda x: float(x), list_)
        dining.append(list_)

for f in os.listdir(running_path):
    for line in open(running_path + f, 'r'):
        line = line.strip()
        list_ = line.split(',')
        del list_[0]
        list_ = map(lambda x: float(x), list_)
        running.append(list_)

dining = filter(lambda x: len(x) == 4, dining)
running = filter(lambda x: len(x) == 4, running)
y_dining = [1]*len(dining)
y_running = [0]*len(running)
X = dining + running
y = y_dining + y_running

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf = svm.SVC()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print score
y_pred = clf.predict(X_test)
print confusion_matrix(y_test, y_pred)
