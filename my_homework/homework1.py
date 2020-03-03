import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
#   Pandas.head()输出前五行数据
#   print(data.head())

#   输出 count mean min等等数据
#   print(data.describe())

'''
    kind：控制图的形状
    x,y：横纵坐标
    figsize：图片大小
'''

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
#   plt.show()

#   在训练集中添加全为1的一列
data.insert(0, 'ones', 1)

#   data.shape[0]返回行数，[1]返回列数
cols = data.shape[1]
#   x是所有行去掉最后一列
x = data.iloc[:, 0:cols-1]
#   y是所有行只有最后一列即label，真正的值
y = data.iloc[:, cols-1:cols]
print(x.head())
print(y.head())

#   将x和y转化为矩阵
x = np.matrix(x.values)
y = np.matrix(y.values)
#   theta是几维取决于x的维度，有多少特征
theta = np.matrix(np.array([0, 0]))

'''
线性回归代价公式:
$$J\left( \theta \right)=\frac{1}{2m}\sum\limits_{i=1}^{m}{{{\left( {{h}{\theta }}\left( {{x}^{(i)}} \right)-{{y}^{(i)}} \right)}^{2}}}$$ 其中：\[{{h}{\theta }}\left( x \right)={{\theta }^{T}}X={{\theta }{0}}{{x}{0}}+{{\theta }{1}}{{x}{1}}+{{\theta }{2}}{{x}{2}}+...+{{\theta }{n}}{{x}{n}}\]
'''


def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


print(computeCost(x, y, theta))


'''
    alpha：学习率
    iters：迭代次数
    
'''
def gradientDescent(X, y, theta, alpha, iters):
    #   temp全零
    temp = np.matrix(np.zeros(theta.shape))
    #   theta的列数
    parameters = int(theta.ravel().shape[1])
    #   记录每次迭代算下来的cost
    cost = np.zeros(iters)

    for i in range(iters):
        #   h(xi)(=theta^T * x)-y
        error = (X * theta.T) - y
    #   公式即可得
        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost

alpha = 0.01
iters = 1000

g,cost = gradientDescent(x, y, theta, alpha, iters)
print(g)
print(computeCost(x, y, g))

#   绘制线性模型
#   linspace 产生x1x2之间的矢量 100为采样点个数
X = np.linspace(data.Population.min(), data.Population.max(), 100)
#   线性的函数
f = g[0, 0] + (g[0, 1] * X)

#   绘制线性的模型
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(X, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

#   绘制迭代的向量
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()



#   处理第二步  多变量

path = 'ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])

#   归一化
data2 = (data2 - data2.mean()) / data2.std()
data2.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]

# convert to matrices and initialize theta
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))

# perform linear regression on the data set
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)

# get the cost (error) of the model
print(computeCost(X2, y2, g2))


#   我们也可以使用scikit-learn的线性回归函数，而不是从头开始实现这些算法。
#   我们将scikit-learn的线性回归算法应用于第1部分的数据，并看看它的表现。

from sklearn import linear_model
model = linear_model.LinearRegression(True,True,1,False)
model.fit(x, y)

X = np.array(x[:, 1].A1)
f = model.predict(x).flatten()

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(X, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()