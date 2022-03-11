import pandas as pd
import matplotlib.pyplot as plt

# a)

miasta = pd.read_csv('Lab01/miasta.csv')

print(miasta)

print(miasta.values)

# b)

newRow = pd.DataFrame([[2010, 460, 555, 405]], columns=['Rok', 'Gdansk', 'Poznan', 'Szczecin'])

miasta = pd.concat([miasta, newRow], ignore_index=True)

print(miasta)

# c)

miasta.plot(x='Rok',
            y='Gdansk',
            color='red',
            marker='o',
            legend=False,
            title='Ludność w miastach Polski'
            #xlabel='Lata'
            #ylabel='Liczba ludności (w tys)'
            )

plt.xlabel('Lata')
plt.ylabel('Liczba ludności (w tys)')

# d)

plt.figure()
plt.plot(miasta.Rok, miasta.Gdansk, label='Gdansk', marker='o', color='red')
plt.plot(miasta.Rok, miasta.Poznan, label='Poznan', marker='o')
plt.plot(miasta.Rok, miasta.Szczecin, label='Szczecin', marker='o', color='green')

plt.legend()
plt.title('Ludność w miastach Polski')
plt.xlabel('Lata')
plt.ylabel('Liczba ludności (w tys)')

plt.show()