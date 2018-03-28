import numpy as np 
import random
# gamma = 0.02
x = np.array([[1,2],[3,4],[5,6]])
# # print(x*x)
z = x
# xx = np.sum(x * x, axis=1)
# print(xx)
# zz = np.sum(z * z, axis=1)
# res = - 2.0 * np.dot(x, z.T) + \
# xx.reshape(-1, 1) + \
# zz.reshape(1, -1)
# print(res)
# k = np.exp(-gamma * res)
print(np.sum(x,axis=1))