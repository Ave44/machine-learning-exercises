import pygad
import numpy

verticalM = [[3,2],[1,3,2],[2,1,2,1],[2,1,1],[2,1],[1,2,1],[6,6],[11],[6,6],[1,1,1],[1,1,1],[1,1],[1,1,2,1],[1,1,2],[3]]
horizontalM = [[1,1],[1,1],[1,1],[1,1],[1,3],[3,3],[1,2,3,3],[1,1,2,1,1,1],[1,5,1,1],[2,3,1],[2,3,1],[1,3,1],[1,1,3,1,1],[2,5,1],[4,1,4]]

gene_space = [0, 1]

def gradeRow(row, pattern):
    rowPattern = []
    for i in row:
        if i == 0:
            rowPattern.append([])
        else:
            rowPattern[-1] = rowPattern[-1] + 1
    return rowPattern


#correct
print(gradeRow([0,1,1,1,0,0,0,1,0,1,1], [3,1,2]))
print(gradeRow([0,1,1,1,0,1,0,0,0,1,1], [3,1,2]))
print(gradeRow([1,1,1,0,0,0,0,1,0,1,1], [3,1,2]))
#zła kolejność bloków
print(gradeRow([0,1,1,1,0,0,1,1,0,0,1], [3,1,2]))
#złe
print(gradeRow([0,1,0,1,0,0,0,1,0,1,1], [3,1,2]))
print(gradeRow([0,1,1,1,1,0,0,1,0,1,1], [3,1,2]))

# def fitness_func(solution, solution_idx):
#     sum1 = numpy.sum(solution * S)
#     solution_invert = 1 - solution
#     sum2 = numpy.sum(solution_invert * S)
#     fitness = -numpy.abs(sum1-sum2)

#     return fitness

# fitness_function = fitness_func
# sol_per_pop = 10

# num_genes = len(verticalM)*len(horizontalM)

# num_parents_mating = 5
# num_generations = 100
# keep_parents = 2
# parent_selection_type = "sss"
# crossover_type = "single_point"
# mutation_type = "random"
# mutation_percent_genes = 8

# # Kryterium zatrzymania stop_criteria=["reach_127.4", "saturate_15"] reach odnosi się do wartości wyniku a saturate do zmian wyniku
# stop_criteria = "reach_0"

# def test():
#     ga_instance = pygad.GA(gene_space=gene_space,
#                        num_generations=num_generations,
#                        num_parents_mating=num_parents_mating,
#                        fitness_func=fitness_function,
#                        sol_per_pop=sol_per_pop,
#                        num_genes=num_genes,
#                        parent_selection_type=parent_selection_type,
#                        keep_parents=keep_parents,
#                        crossover_type=crossover_type,
#                        mutation_type=mutation_type,
#                        mutation_percent_genes=mutation_percent_genes,
#                        stop_criteria=stop_criteria)
#     ga_instance.run()
#     print("Generations:", ga_instance.generations_completed, "Fitness:", ga_instance.best_solution()[1])
#     return ga_instance.generations_completed
