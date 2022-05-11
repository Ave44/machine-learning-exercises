import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# df = pd.read_csv("./Lab08/iris.csv") # dla Windows'a
df = pd.read_csv("iris.csv") # dla Linux'a

df['class'] = df[['class']].replace(['setosa', 'versicolor', 'virginica'], [0, 1, 2])

feature_cols = ['sepallength', 'sepalwidth', 'petallength', 'petalwidth']
features = df[feature_cols]
labels = df['class']
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3, random_state=10)

def runClassifier(hidden_layer_sizes):
    mlp = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes=hidden_layer_sizes,
                        random_state=1)

    mlp.fit(train_features, train_labels)

    print("Poprawność klayfikatora: ", mlp.score(test_features, test_labels))

    # Macierz błędu
    class_true = []
    class_pred = mlp.predict(test_features)

    for i in range(len(test_features)):
        class_true.append(test_labels.iloc[i])

    labelNames = ['setosa', 'versicolor', 'virginica']
    labels = [0, 1, 2]
    print('\nConfusion matrix:\n')
    matrix = confusion_matrix(class_true, class_pred, labels=labels)

    print("           | %-10s | %-10s | %-10s" % (labelNames[labels[0]], labelNames[labels[1]], labelNames[labels[2]]))
    print("----------------------------------------------------")
    for i in range(len(labels)):
        print("%-10s | %6d     | %6d     | %6d" % (labels[i], matrix[i][0], matrix[i][1], matrix[i][2]))
        print("----------------------------------------------------")


runClassifier((2, 1))
runClassifier((3, 1))
runClassifier((3, 3, 1))
runClassifier((3, 3))
