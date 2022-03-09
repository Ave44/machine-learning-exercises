def prime(n):
    if (n < 2):
        return False
    lista = []
    for i in range(int((n-1)/2)):
        lista.append(i+2)
    while len(lista) != 0:
        x = lista[0]
        if(n % x == 0):
            return False
        lista.remove(x)
        lista = removeMultipies(x, lista)
    return True

def removeMultipies(n, lista):
    nowaLista = []
    for i in lista:
        if(i % n != 0):
            nowaLista.append(i)
    return nowaLista

print(1,prime(1))
print(2,prime(2))
print(3,prime(3))
print(4,prime(4))
print(5,prime(5))
print(6,prime(6))
print(11,prime(11))

def select_primes(lista):
    pierwsze = []
    for i in lista:
        if(prime(i)):
            pierwsze.append(i)
    return pierwsze

print(select_primes([1,2,3,4,5,6,7,8,9,10,11,12,17,19,20,21,49]))