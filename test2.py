import scipy.io as sio
import numpy as np 
import random
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
# f = sio.loadmat('f:\\matlab\Hw2-package\spamTrain.mat')
# ff = sio.loadmat('f:\\matlab\Hw2-package\spamTest.mat')
# X = f['X']
# y = f['y']
# testx = ff['Xtest']
# testy = ff['ytest'].reshape(1,-1)[0].astype(int)

f = open('iris-virginica.txt', 'r')
TrainingX = []
TrainingY = []
TestX = []
TestY = []
for i in f.readlines():
	line = i.split(',')
	l = []
	for j in range(4):
		l.append(float(line[j]))
	if random.random() < 0.7:
		TrainingX.append(l)
		TrainingY.append(int(line[-1].strip('\n')))
	else:
		TestX.append(l)
		TestY.append(int(line[-1].strip('\n')))
TrainingX = np.array(TrainingX)
TrainingY = np.array(TrainingY)
TestX = np.array(TestX)
TestY = np.array(TestY)

# print(TrainingX)
# y = y.reshape(1,-1)[0].astype(int)
# print(xtest)
# s = BernoulliNB()
s = svm.SVC(kernel = 'rbf')
s.fit(TrainingX,TrainingY)
y_ = s.predict(TestX)
count = 0
for i in range(len(TestY)):
    if TestY[i] == y_[i]:
        count += 1

print(count/len(TestY))