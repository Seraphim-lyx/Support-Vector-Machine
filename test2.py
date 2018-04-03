import scipy.io as sio
import numpy as np 
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
f = sio.loadmat('f:\\matlab\Hw2-package\spamTrain.mat')
ff = sio.loadmat('f:\\matlab\Hw2-package\spamTest.mat')
X = f['X']
y = f['y']
testx = ff['Xtest']
testy = ff['ytest'].reshape(1,-1)[0].astype(int)



y = y.reshape(1,-1)[0].astype(int)
# print(xtest)
s = BernoulliNB()
# s = svm.SVC(kernel = 'poly')
s.fit(X,y)
y_ = s.predict(testx)
count = 0
for i in range(len(testy)):
    if testy[i] == y_[i]:
        count += 1

print(count/len(testy))