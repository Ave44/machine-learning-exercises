import pygad
import numpy
import time

S = [1, 2, 3, 6, 10, 17, 25, 29, 30, 41, 51, 60, 70, 79, 80]

gene_space = [0, 1]

def fitness_func(solution, solution_idx):
    sum1 = numpy.sum(solution * S)
    solution_invert = 1 - solution
    sum2 = numpy.sum(solution_invert * S)
    fitness = -numpy.abs(sum1-sum2)
    #lub: fitness = 1.0 / (1.0 + numpy.abs(sum1-sum2))
    return fitness

fitness_function = fitness_func
sol_per_pop = 10
num_genes = len(S)
num_parents_mating = 5
num_generations = 30
keep_parents = 2
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 8

# Kryterium zatrzymania stop_criteria=["reach_127.4", "saturate_15"] reach odnosi się do wartości wyniku a saturate do zmian wyniku
stop_criteria = "reach_0"

def test():
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
    print("Generations:", ga_instance.generations_completed, "Fitness:", ga_instance.best_solution()[1])
    return ga_instance.generations_completed

timeArray = []

for i in range(10):
    start = time.time()
    test()
    end = time.time()
    timeArray.append(end - start)

res = 0
for i in timeArray:
    res += i

print("Average: ", res/len(timeArray))