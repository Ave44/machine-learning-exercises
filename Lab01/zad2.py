import random
import math

a = [3, 8, 9, 10, 12]
b = [8, 7, 7, 5, 6]

# a)
def sum(a, b):
    vector = []
    for i in range(len(a)):
        vector.append(a[i]+b[i])
    return vector

print('\nsum', sum(a, b))

def iloczyn(a, b):
    vector = []
    for i in range(len(a)):
        vector.append(a[i]*b[i])
    return vector

print('\niloczyn', iloczyn(a, b))

# b)
def iloczyn_skalarny(a, b):
    result = 0
    for i in range(len(a)):
        result = result + a[i]*b[i]
    return result

print('\niloczyn skalarny', iloczyn_skalarny(a, b))

# c)
def eukl(vector):
    result = 0
    for i in range(len(a)):
        result = result + a[i]*a[i]
    return math.sqrt(result)

print('\neukl', eukl(a))

# d)

def create_rand_vector(dim):
    vector = []
    for i in range(dim):
        vector.append(random.randint(1, 100))
    return vector

vec = create_rand_vector(50)
print('\nvector', vec)

# e)

def average(vec):
    sum = 0
    for i in vec:
        sum += i
    return sum/len(vec)

def minimum(vec):
    min = vec[0]
    for i in vec:
        if(i < min):
            min = i
    return min

def maximum(vec):
    max = vec[0]
    for i in vec:
        if(i > max):
            max = i
    return max

def standard_deviation(vec):
    avg = average(vec)
    sum = 0
    for i in vec:
        sum += pow(i-avg, 2)
    return math.sqrt(sum/(len(vec)))

print('\navg', average(vec))
print('\nmin', minimum(vec))
print('\nmax', maximum(vec))
print('\nsd', standard_deviation(vec))

# f)

def normalize(vec):
    min = minimum(vec)
    max = maximum(vec)
    normalized_vec = []
    for i in vec:
        normalized_vec.append((i-min)/(max-min))
    return normalized_vec

print('\nnormalize', normalize(vec)) # na miejscu max stoi 1, a min 0

# g)

def standarize(vec):
    avg = average(vec)
    sd = standard_deviation(vec)
    stanarized_vec = []
    for i in vec:
        stanarized_vec.append((i-avg)/sd)
        print(i, avg, sd, (i-avg)/sd)
    return stanarized_vec

z = standarize(vec)
print('\nstandarize', z) 
print('\nstandarize avg', average(z)) # 0
print('\nstandarize sd', standard_deviation(z))  # 1

# h)

def discretize(vec):
    discretized_vec = []
    for i in vec:
        discretized_vec.append(get_range(i))
    return discretized_vec

def get_range(n):
    ranges = [10,20,30,40,50,60,70,80,90]
    for i in ranges:
        if(n < i):
            return '[' + str(i-10) + ', ' + str(i) + ')'
    return '[90, 100]'

print('discretize', discretize(vec))