# coding: utf-8


from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


# 这里是具体怎么实现的

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

knn = KNeighborsClassifier(n_neighbors=5)  # 选择 k 为 5
knn.fit(X_train, y_train)

print knn.score(X_test, y_test)

# 0.973684210526  这个数字还是比较理想的
# 上面是人为的用 train_test_split 拆分的训练数据和测试数据，为了实现不同分组交叉验证，我们使用如下方法：

scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')  # cv = 5 将数据分成五份交叉验证

print scores
print scores.mean()

# 每组分数为：[ 0.96666667  1.          0.93333333  0.96666667  1.        ]
# 平均分数为：0.973333333333


# 这里选择不同的 k 来验证最终的分数，从而确定哪个 k 最合适

score_li = []
k_range = range(1, 31)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')  # cv = 5 将数据分成五份交叉验证，classification 用 accuracy 比较好
    score_li.append(scores.mean())

loss_li = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    loss = -cross_val_score(knn, X, y, cv=5, scoring='neg_mean_squared_error')  # cv = 5 将数据分成五份交叉验证，regression 用 mean_squared_error 比较好
    loss_li.append(loss.mean())

plt.plot(k_range, loss_li)
plt.show()
