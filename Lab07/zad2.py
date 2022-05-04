import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("iris.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=13)
print(train_set[train_set[:, 4].argsort()]) # wyświetlanie posortowanych wartości

def classify_iris(sl, sw, pl, pw):
    if sl > 4:
        return "setosa"
    elif pl <= 5:
        return "virginica"
    else:
        return "versicolor"

good_predictions = 0
len = test_set.shape[0]

for i in range(len):
    if classify_iris(test_set[i][0], test_set[i][1], test_set[i][2], test_set[i][3]) == test_set[i][4]:
        good_predictions = good_predictions + 1
print(good_predictions)
print(good_predictions / len * 100, "%")

# print(train_set)

def classify_irisV2(sl, sw, pl, pw): # Pełna skuteczność
    if pw < 0.8:
        return "setosa"
    elif pw >= 1.8 or sl > 7 or pl > 5 or sw > 3.5 and sl > 5.6:
        return "virginica"
    else:
        return "versicolor"

good_predictions = 0

for i in range(len):
    if classify_irisV2(test_set[i][0], test_set[i][1], test_set[i][2], test_set[i][3]) == test_set[i][4]:
        good_predictions = good_predictions + 1

print("Version 2")
print(good_predictions)
print(good_predictions / len * 100, "%")