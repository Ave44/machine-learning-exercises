import pygad
import math
import time

labirynt = [[1,1,1,1,1,1,1,1,1,1,1,1],
            [1,2,0,0,1,0,0,0,1,0,0,1],
            [1,1,1,0,0,0,1,0,1,1,0,1],
            [1,0,0,0,1,0,1,0,0,0,0,1],
            [1,0,1,0,1,1,0,0,1,1,0,1],
            [1,0,0,1,1,0,0,0,1,0,0,1],
            [1,0,0,0,0,0,1,0,0,0,1,1],
            [1,0,1,0,0,1,1,0,1,0,0,1],
            [1,0,1,1,1,0,0,0,1,1,0,1],
            [1,0,1,0,1,1,0,1,0,1,0,1],
            [1,0,1,0,0,0,0,0,0,0,3,1],
            [1,1,1,1,1,1,1,1,1,1,1,1]]

for i in labirynt:
    for j in i:
        if j == 1:
            print('██', end='')
        elif j == 0:
            print('  ', end='')
        else:
            print('▒▒', end='')
    print('')

directions = {1: '↑', 2: '→', 3: '↓', 4: '←'}

def distance (x, y, destX=10, destY=10):
    return math.sqrt(abs(destX - x)**2 + abs(destY - y)**2)

def fitness_function (solution, solution_idx):
    x = 1
    y = 1
    for i in solution:
        if i == 1:
            if labirynt[y-1][x] == 1:
                break
            y -= 1
        if i == 2:
            if labirynt[y][x+1] == 1:
                break
            x += 1
        if i == 3:
            if labirynt[y+1][x] == 1:
                break
            y += 1
        if i == 4:
            if labirynt[y][x-1] == 1:
                break
            x -=1
        if distance(x, y) == 0:
            return 0
    
    return (1/distance(x, y)) - 2

gene_space = [1, 2, 3, 4]
num_genes = 30
num_generations = 500
stop_criteria = "reach_0"

sol_per_pop = 40
keep_parents = 4
num_parents_mating = 18
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 4

def opposite (a, b):
    if abs(a - b) % 2 == 0 and a != b:
        return True
    return False

def clearResult (solution):
    test = True
    result = []
    i = 0

    while i < len(solution):
        if len(result) == 0:
            if i < len(solution)-1:
                if not opposite(solution[i], solution[i+1]):
                    result.append(solution[i])
                    i += 1
                else:
                    i += 2
            else:
                result.append(solution[i])
                i += 1
        else:
            if not opposite(result[-1], solution[i]):
                result.append(solution[i])
                i += 1
            else:
                result.pop()
                i += 1

    if test:
        cuted = []
        x = 1
        y = 1
        for i in result:
            if distance(x, y) == 0:
                return cuted
            if i == 1:
                y -= 1
            if i == 2:
                x += 1
            if i == 3:
                y += 1
            if i == 4:
                x -=1
            cuted.append(i)
        return cuted
    return clearResult(result)

def runEvolution():
    ga_instance = pygad.GA(gene_space=gene_space,
                        num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        fitness_func=fitness_function,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
                        parent_selection_type=parent_selection_type,
                        keep_parents=keep_parents,
                        crossover_type=crossover_type,
                        mutation_type=mutation_type,
                        mutation_percent_genes=mutation_percent_genes,
                        stop_criteria=stop_criteria)

    ga_instance.run()
    return ga_instance
    # printResult(ga_instance)

def printResult(ga_instance):
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Generations:", ga_instance.generations_completed)
    print("Solution:\t", end=" ")
    for i in solution:
        print(directions[i], end=" ")
    print("\nSolution fitness: ", solution_fitness)

    cleanSolution = clearResult(solution)
    print("Clean solution:\t", end=" ")
    for i in cleanSolution:
        print(directions[i], end=" ")
    print("\nClean solution length:", len(cleanSolution))
    
    ga_instance.plot_fitness()

printResult(runEvolution())

timeArray = []
genArray = []

for i in range(10):
    print('.', end='', flush=True)
    start = time.time()
    ga_instance = runEvolution()
    end = time.time()
    timeArray.append(end - start)
    genArray.append(ga_instance.generations_completed)

timeRes = 0
genRes = 0
for i in range(len(timeArray)):
    timeRes += timeArray[i]
    genRes += genArray[i]

print("\nAverage time: \t", timeRes/len(timeArray))
print("Average generations: \t", genRes/len(genArray))