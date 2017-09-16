import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout
import os
import matplotlib.pyplot as plt

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
for i in range(len(dining)):
    if dining[i][3] < 0:
        dining[i][3] += 256
dining = filter(lambda x: x[3]>=60, dining)

running = filter(lambda x: len(x) == 4, running)
for i in range(len(running)):
    if running[i][3] < 0:
        running[i][3] += 256
#plt.plot(running)
#plt.show()
running = filter(lambda x: x[3]>=60, running)

X_dining = []
X_running = []

i = 0
while(i <= len(dining) and i+149 <= len(dining)):
    tmp = []
    for j in range(0, 150):
        tmp = tmp + dining[i+j]
    #if i == 0:
    #    print tmp
    #    print len(tmp)
    X_dining.append(tmp)
    i += 150

i = 0
while(i <= len(running) and i+149 <= len(running)):
    tmp = []
    for j in range(0, 150):
        tmp = tmp + dining[i+j]
    X_running.append(tmp)
    i += 150


y_dining = [1]*len(X_dining)
y_running = [0]*len(X_running)
X = X_dining + X_running
y = y_dining + y_running

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
#input layer
model.add(Dense(300, activation='relu', input_shape=(600,)))
#hidden layer
model.add(Dropout(0.2))
model.add(Dense(150, activation='relu'))
#output layer
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=1)

#y_pred = model.predict(X_test)
score = model.evaluate(X_test, y_test, verbose=1)
print score
