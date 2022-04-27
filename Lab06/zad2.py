import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("./Lab06/iris.csv")
print("\n=== Początkowe dane ===\n", df, sep="")
print("= variancje =")
print(np.var(df['sepallength']),np.var(df['sepalwidth']),np.var(df['petallength']),np.var(df['petalwidth']))
# 0.6811222222222223 0.18675066666666668 3.092424888888889 0.5785315555555555
print("\n======================================================================")



# standaryzacja
features = ['sepallength', 'sepalwidth', 'petallength', 'petalwidth']
x = df.loc[:, features].values
y = df.loc[:,['class']].values
x = StandardScaler().fit_transform(x)

standarizedDf = pd.concat([pd.DataFrame(x, columns=features), df[['class']]], axis = 1)
print("\n=== Ustandaryzowane dane ===\n", standarizedDf, sep="")
print("= variancje =")
print(np.var(standarizedDf['sepallength']),np.var(standarizedDf['sepalwidth']),np.var(standarizedDf['petallength']),np.var(standarizedDf['petalwidth']))
# 1.0 0.9999999999999998 0.9999999999999997 1.0000000000000002
print("\n======================================================================")



# wykonanie PCA dla 2 kolumn
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df[['class']]], axis = 1)

print("\n=== Finalne dane - 2 kolumny===\n", finalDf, sep="")

print("Zawartość informacji w danych kolumnach: ", pca.explained_variance_ratio_)
print("Strata informacji po usunięciu kolumn: ", 1-sum(pca.explained_variance_ratio_))
print("\n======================================================================")

# # wyświetlanie wykresu dwuwymiarowedgo
# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1) 
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 component PCA', fontsize = 20)
# targets = ['setosa', 'versicolor', 'virginica']
# colors = ['r', 'g', 'b']
# for target, color in zip(targets,colors):
#     indicesToKeep = finalDf['class'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#                , finalDf.loc[indicesToKeep, 'principal component 2']
#                , c = color
#                , s = 50)
# ax.legend(targets)



# wykonanie PCA dla 3 kolumn
pca2 = PCA(n_components=3)
principalComponents2 = pca2.fit_transform(x)
principalDf2 = pd.DataFrame(data = principalComponents2, columns = ['component 1', 'component 2', 'component 3'])
finalDf2 = pd.concat([principalDf2, df[['class']]], axis = 1)

print("\n=== Finalne dane - 3 kolumny===\n", finalDf2, sep="")

print("Zawartość informacji w danych kolumnach: ", pca2.explained_variance_ratio_)
print("Strata informacji po usunięciu kolumn: ", 1-sum(pca2.explained_variance_ratio_))
print("\n======================================================================")

# # wyświetlanie wykresów trójwymiarowego
# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(projection='3d') 
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_zlabel('Principal Component 3', fontsize = 15)
# ax.set_title('3 component PCA', fontsize = 20)
# targets = ['setosa', 'versicolor', 'virginica']
# colors = ['r', 'g', 'b']
# for target, color in zip(targets,colors):
#     indicesToKeep = finalDf2['class'] == target
#     ax.scatter(finalDf2.loc[indicesToKeep, 'component 1']
#                , finalDf2.loc[indicesToKeep, 'component 2']
#                , finalDf2.loc[indicesToKeep, 'component 3']
#                , c = color
#                , s = 50)
# ax.legend(targets)

# wyświetlanie obu wykresów
# 2D
plt.figure(figsize = (12,6))
plt.subplot(1,2,1)
plt.xlabel('Principal Component 1', fontsize = 10)
plt.ylabel('Principal Component 2', fontsize = 10)
plt.title('2 component PCA', fontsize = 20)
targets = ['setosa', 'versicolor', 'virginica']
colors = ['#cf2a00', '#3dad00', '#005e94']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['class'] == target
    plt.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
plt.legend(targets)
# 3D
ax = plt.subplot(1,2,2, projection='3d') 
ax.set_xlabel('Principal Component 1', fontsize = 10)
ax.set_ylabel('Principal Component 2', fontsize = 10)
ax.set_zlabel('Principal Component 3', fontsize = 10)
plt.title('3 component PCA', fontsize = 20)
for target, color in zip(targets,colors):
    indicesToKeep = finalDf2['class'] == target
    ax.scatter(finalDf2.loc[indicesToKeep, 'component 1']
               , finalDf2.loc[indicesToKeep, 'component 2']
               , finalDf2.loc[indicesToKeep, 'component 3']
               , c = color
               , s = 50)
ax.view_init(20, -110)
plt.legend(targets)
plt.show()