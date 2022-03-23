import pygad

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

gene_space = [1,2,3,4]
num_generations = 100
num_parents_mating = 10
fitness_func = ()


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