import numpy as np
from sklearn.svm import SVC

train_set = np.loadtxt('dota2Train.csv', delimiter=',')
test_set = np.loadtxt('dota2Test.csv', delimiter=',')

X = train_set[:, 1:]
y = train_set[:, 0]

test_X = test_set[:, 1:]
test_y = test_set[:, 0]

clf = SVC()
clf.fit(X, y)

predicted = clf.predict(test_X)

wrong_num = sum(1 for i, j in zip(predicted, test_y) if i != j)

print wrong_num/10294.0
