import pygad
import math

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

    return 1/distance(x, y) - 1

gene_space = [1, 2, 3, 4]
num_genes = 30
num_generations = 500
# zazwyczaj algorytm kończy po 50-250 generacjach (specjalnie ustawiłem zawyżoną wartość
# żeby zmaksymalizować prawdopodobieństwo odnaleziena wyniku za każdym uruchomieniem algorytmu,
# ponieważ wszystkie nadmiarowe generacje i tak zostają zatrzymane przez stop_criteria)
stop_criteria = "reach_0"

sol_per_pop = 40
keep_parents = 4
num_parents_mating = 18
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 5 # najmiejsza możliwa wartość to 4 ale gdy jest ustawione na 5 algorytm działa lepiej

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

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Generations:", ga_instance.generations_completed)
print("Solution:\t", end=" ")
for i in solution:
    print(directions[i], end=" ")
print("\nSolution fitness: ", solution_fitness)

def opposite (a, b):
    if abs(a - b) % 2 == 0 and a != b:
        return True
    return False

def clearResult (solution):
    test = True
    result = []
    i = 0
    while i <= len(solution):
        if i == len(solution)-1 and not opposite(solution[i], result[-1]):
            result.append(solution[i])
            break
        elif i == len(solution):
            break
        else:
            if opposite(solution[i], solution[i+1]):
                i += 2
                test = False
            else:
                result.append(solution[i])
                i += 1
    if test:
        return result
    return clearResult(result)

cleanSolution = clearResult(solution)
print("Clean solution:\t", end=" ")
for i in cleanSolution:
    print(directions[i], end=" ")
print("\nClean solution length:", len(cleanSolution))

# ga_instance.plot_fitness() ###################       ODKOMENTOWAĆ         ################################