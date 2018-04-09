import numpy as np
import random
import scipy.io as sio
from sklearn import svm

class SMO(object):
    """
    simple implementation of support vector machine
    based on sequential minimal optimization algorithm
    optional kernel: linear, gaussian(rbf), polynomial(poly)
    X -> two dimension data set
    y -> one dimension data set
     """

    def __init__(self, kernel='linear', degree=2, C=1, gamma=None, tol=1e-3):
        self.kernel = kernel
        self.gamma = gamma  # parameter for gaussian
        self.C = C  # regularization
        self.tol = tol  # tolerance edge
        self.degree = degree  # power degree for polynomial kernel

    def _kernel(self, x, z=None):
        if z is None:
            z = np.copy(x)
        if self.kernel == 'linear':
            return self.linearKernel(x, z)
        elif self.kernel == 'poly':
            return self.polyKernel(x, z)
        elif self.kernel == 'rbf':  # K(x,z)=exp(− gamma*|x−z|**2)
            return self.GaussianKernel(x, z)
        else:
            print("kernel error")

    def train(self, X, y):
        # X = np.array(X)
        # y = np.array(y)
        self.X = X
        self.y = y
        k = self._kernel(X)
        # print(k)
        self.alpha = np.zeros(len(X))
        self.E = np.zeros(len(X))
        self.b = 0.0

        count = 0
        while count < 5:
            # print(count)
            alpha_change = 0
            for i in range(len(self.alpha)):
                # print(alpha_change)
                # Calculating error for i
                self.E[i] = self.b + np.sum(self.alpha * y * k[i]) - y[i]
                if((self.E[i] * y[i] < - self.tol) and (self.alpha[i] < self.C)) or \
                  ((self.E[i] * y[i] > self.tol) and (self.alpha[i] > 0)):
                    # print('inside')
                    # j = random.randint(0,len(self.alpha)-1)
                    j = np.round(random.random() * len(y) - 1).astype(int)
                    while i == j:
                        j = np.round(random.random() * len(y) - 1).astype(int)
                        # j = random.randint(0, len(self.alpha)-1)
                    # Calculating error for j
                    self.E[j] = self.b + np.sum(self.alpha * y * k[j]) - y[j]

                    ai = self.alpha[i]  # old alpha i
                    aj = self.alpha[j]  # old alpha j

                    # calculating bounding area for alpha
                    if y[i] == y[j]:
                        L = max(0, self.alpha[j] + self.alpha[i] - self.C)
                        H = min(self.C, self.alpha[j] + self.alpha[i])
                    else:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])

                    if L == H:
                        # next iteration if no area
                        # print("L==H")
                        continue

                    eta = 2 * k[i, j] - k[i, i] - k[j, j]
                    if eta >= 0:
                        # print("eta out")
                        continue

                    # clip new value
                    self.alpha[j] = self.alpha[j] - \
                        (y[j] * (self.E[i] - self.E[j])) / eta

                    # bound new value
                    self.alpha[j] = min(H, self.alpha[j])
                    self.alpha[j] = max(L, self.alpha[j])

                    if abs(self.alpha[j] - aj) < self.tol:
                        # if change less than tol
                        self.alpha[j] = aj
                        # print("too small")
                        continue

                    # tune alpha i accroding to alpha j
                    self.alpha[i] = ai + y[i] * y[j] * (aj - self.alpha[j])

                    bi = self.b - self.E[i] - y[i] * (self.alpha[i] - ai) * k[i, i] + \
                        y[j] * (self.alpha[j] - aj) * k[i, j]

                    bj = self.b - self.E[j] - y[i] * (self.alpha[i] - ai) * k[i, j] + \
                        y[j] * (self.alpha[j] - aj) * k[j, j]

                    if 0 < self.alpha[i] and self.alpha[i] < self.C:
                        self.b = bi
                    elif 0 < self.alpha[j] and self.alpha[j] < self.C:
                        self.b = bj
                    else:
                        self.b = (bi + bj) / 2.0

                    alpha_change += 1

                    # print("iter:{0}, change:{1}".format(i, alpha_change))

            # iteration until alpha converge

            if alpha_change == 0:
                # print("count + 1")
                count += 1
            else:
                # print("reset count")
                count = 0
            print("whole iter number {0}".format(count))
        # index = np.where(self.alpha < 0)[0]
        # self.alpha[index] = 0
        self.w = np.dot(self.alpha * y, X)
        index = np.where(self.alpha > 0)[0]
        self.alpha = self.alpha[self.alpha > 0]
        self.X = self.X[index]
        self.y = self.y[index]


    def predict(self, px):
        p = np.array([])
        if self.kernel == 'linear':
            p = np.dot(self.w, px.T) + self.b

        elif self.kernel == 'rbf':
            p = self._kernel(px, self.X)
            p = p * self.alpha
            p = p * self.y
            p = np.sum(p, axis=1)
        else:
            p = np.dot(self.alpha * self.y, self._kernel(self.X, px)) + self.b
        return np.sign(p)

    def evaluate(self, ry, py):
        hit = 0
        for i in range(len(py)):
            if py[i] == ry[i]:
                hit += 1
        return hit / len(py)

    def linearKernel(self, x, z):
        # print(np.dot(x, z.T))
        return np.dot(x, z.T)

    def polyKernel(self, x, z):
        return np.power(np.dot(x, z.T + 1.0), 2)

    def GaussianKernel(self, x, z):

        if self.gamma is None:
            self.gamma = 1 / np.shape(x)[1]

        xx = np.sum(x * x, axis=1)
        zz = np.sum(z * z, axis=1)
        res = - 2.0 * np.dot(x, z.T) + \
            xx.reshape(-1, 1) + \
            zz.reshape(1, -1)
        # value = (-1*1) / (2 * np.power (2,2))
        # exp = np.exp(value)
        # return exp**res # 0 < gamma < 1
        return np.exp(-1.0 * self.gamma * res)


# f = sio.loadmat('f:\\matlab\Hw2-package\spamTrain.mat')
# ff = sio.loadmat('f:\\matlab\Hw2-package\spamTest.mat')
# XX = f['X'][:2000].astype(int)
# testx = ff['Xtest'].astype(int)
# yy = f['y'][:2000].astype(int)
# testy = ff['ytest'].reshape(1, -1)[0].astype(int)
# yy = yy.reshape(1, -1)[0].astype(int)
# yy[yy == 0] = -1
# testy[testy == 0] = -1

# smo = SMO(kernel='linear')
# smo.train(XX, yy)
# print(len(smo.alpha))
# p = smo.predict(testx)


# print(smo.evaluate(testy, p))
# 

f = open('output1.txt', 'w')
label = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
x = np.loadtxt('a.txt', delimiter = ",").astype(int)
y_ = np.loadtxt('label.txt',delimiter = ",").astype(int)
y_[y_==0] = -1

y = y_.T[5].astype(int)






TrainingX = x[:4000]
TrainingY = y[:4000]
TestX = x[4000:]
TestY = y[4000:]

# smo = SMO(kernel='rbf')
# smo.train(TrainingX, TrainingY)

# p = smo.predict(TestX)
# print(p)
# accuracy = smo.evaluate(TestY, p)
# print(accuracy)
    # f.writelines("The accuracy for {0} is {1}\n".format(label[j], accuracy))

# f.close()    

s = svm.SVC(kernel = 'rbf')
s.fit(TrainingX,TrainingY)
y_ = s.predict(TestX)
print(TestX)
print(y_)
count = 0
for i in range(len(TestY)):
    if TestY[i] == y_[i]:
        count += 1

print(count/len(TestY))