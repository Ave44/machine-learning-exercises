import pygad
import numpy

SWag = [7, 7, 6, 2, 5, 6, 1, 3, 10, 3, 15]
SCen = [100, 300, 200, 40, 500, 70, 100, 250, 300, 280, 300]
SNam = ['zegar', 'obraz-pejzaż', 'obraz-portret', 'radio', 'laptop', 'lampka nocna', 'srebrne sztućce', 'porcelana', 'figura z bronzu', 'skóżana torebka', 'odkurzacz']

gene_space = [0, 1]

def fitness_func(solution, solution_idx):
    wag = numpy.sum(solution * SWag)
    if wag > 25:
        return 0
    cen = numpy.sum(solution * SCen)
    fitness = cen
    return fitness

fitness_function = fitness_func

sol_per_pop = 10
num_genes = len(SNam)
num_parents_mating = 5
num_generations = 30
keep_parents = 2
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 10

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
                       mutation_percent_genes=mutation_percent_genes)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Przedmioty : ")
for i in range(len(solution)):
    if solution[i] == 1:
        print(SNam[i])

print("\nFitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

prediction = numpy.sum(SCen*solution)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))
ga_instance.plot_fitness()