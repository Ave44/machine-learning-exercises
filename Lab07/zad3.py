import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree

df = pd.read_csv("iris.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=13)
# # wyświetlanie posortowanych wartości
# print(train_set[train_set[:, 4].argsort()])
# print(test_set[test_set[:, 4].argsort()])

clf = tree.DecisionTreeClassifier()
print(clf)
