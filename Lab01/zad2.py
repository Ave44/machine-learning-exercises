a = [3, 8, 9, 10, 12]
b = [8, 7, 7, 5, 6]

def sum(a, b):
    vector = []
    for i in range(len(a)):
        vector.append(a[i]+b[i])
    return vector

print(sum(a, b))

def iloczyn(a, b):
    vector = []
    for i in range(len(a)):
        vector.append(a[i]*b[i])
    return vector

print(iloczyn(a, b))

def iloczyn_skalarny(a, b):
    result = 0
    for i in range(len(a)):
        result = result + a[i]*b[i]
    return result

print(iloczyn_skalarny(a, b))

def eukl(vector):
    result = 0
    for i in range(len(a)):
        result = result + a[i]*a[i]
    return result

print(eukl(a))