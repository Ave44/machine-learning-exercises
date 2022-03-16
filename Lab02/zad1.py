S1 = [1, 3, 10, 17, 30, 41, 70, 80]
S2 = [2, 6, 25, 29, 51, 60, 79]

def sum(list):
    res = 0
    for i in list:
        res += i
    return res

print(sum(S1))
print(sum(S2))
print(sum(S1) == sum(S2))