import pygad
import numpy as np
import matplotlib.pyplot as plt

verticalM = [[3,2],[1,3,2],[2,1,2,1],[2,1,1],[2,1],[1,2,1],[6,6],[11],[6,6],[1,1,1],[1,1,1],[1,1],[1,1,2,1],[1,1,2],[3]]
horizontalM = [[1,1],[1,1],[1,1],[1,1],[1,3],[3,3],[1,2,3,3],[1,1,2,1,1,1],[1,5,1,1],[2,3,1],[2,3,1],[1,3,1],[1,1,3,1,1],[2,5,1],[4,1,4]]

# verticalM2 = [[1,2],[2,2,4],[3,3,1,5],[1,8,1,5],[8,2,2,3],[4,1,1,2],[1,1,1,1],[2,2,2],[4,1],[5,1],[6],[2,1,2],[2,1],[5,1],[2,1],[1,2,2,1],[2,1,2,1],[5,1,2],[2,1,1],[2,1,1]]
# horizontalM2 = [[3],[2,1],[5],[2,1,2],[2,2,3],[3,3,1],[9,3],[4,4,1,3],[2,2,3,1],[2,3,4],[4],[1,2],[4,3,3],[4,1,1],[4],[3,1,3],[5,1,3],[6,6],[4],[3]]

# verticalM3 = [[3,2],[2,3,1,1],[1,3,1,2,1],[3,1,1,2,2,5],[2,2,1,2,1,9],[2,1,2,2,6,4],[2,2,2,1,1,7,2],[1,3,2,2,9,1],[1,3,2,1,1,10],[1,2,2,1,1,1,4,6],[1,2,2,3,3,5,5],[1,2,2,2,1,1,2,7,4],[1,3,3,2,1,2,9,4],[2,2,5,3,12,3],[2,3,9,3,8,2],[1,2,19,6,2],[2,3,18,5,1],[1,3,9,6,5,1],[2,4,6,5,4,1],[2,6,4,4,4],[5,10,4,3,3],[12,4,3,3,3],[2,10,3,3,3,3],[2,2,9,3,2,3,4],[1,2,6,3,3,3,2,1],[3,1,5,3,2,3,2,1],[1,2,1,4,3,1,3,2,1],[2,1,1,2,2,5,2,2,1],[4,1,2,3,3,5,3,2,2],[4,1,2,6,2,2,3,2,2]]
# horizontalM3 = [[3,10,2,1,2],[2,4,4,2,1,3],[1,3,4,3,1,1,3],[3,9,2,2,2],[3,5,5,2,1,2],[2,2,4,4,1,2,2],[4,3,1,2],[4,2,2,3],[3,1,2,2],[5,3,2,1],[4,3,1,2],[2,6,4],[1,1,3,3,5],[4,1,3,5],[2,3,1,6,5,2],[1,5,4,4],[2,3,4,2],[2,1,3,5,1],[1,3,5,1],[4,1,4,2],[4,4,2,7],[2,2,5,1,5],[3,6,2,3,1],[4,2,3,5],[1,6,3,2],[2,2,3,2,7],[3,3,4,2,6],[9,4,3,4],[11,4,4],[11,4,4],[12,4,3,1],[6,6,5,2],[2,4,6,9],[2,4,6,7],[3,4,6,3,1],[2,5,6,3],[3,5,14],[2,5,10,1],[1,6,5,2],[7,6]]

def createImg(genes, width, height):
    img = np.zeros((height, width))
    index = 0
    for i in range(height):
        for j in range(width):
            img[i][j] = abs(genes[index]-1)
            index += 1
    
    return img

def gradeFunc(solution, pattern):
    solutionPattern = [0]
    for i in solution:
        if i == 0:
            solutionPattern.append(0)
        else:
            if solutionPattern[-1] != 0:
                solutionPattern[-1] = solutionPattern[-1] + 1
            else:
                solutionPattern.append(1)
    filteredRowPatern = list(filter(lambda n: n != 0 , solutionPattern))

    grade = -abs(sum(pattern) - sum(filteredRowPatern))

    for i in range(len(pattern)):
        if i < len(filteredRowPatern):
            if pattern[i] != filteredRowPatern[i]:
                grade -= abs(pattern[i] - filteredRowPatern[i])
        else:
            grade -= pattern[i]

    return grade
    # return [grade, filteredRowPatern, pattern]

# #correct
# print(gradeFunc([0,1,1,1,0,0,0,1,0,1,1], [3,1,2]))
# print(gradeFunc([0,1,1,1,0,1,0,0,0,1,1], [3,1,2]))
# print(gradeFunc([1,1,1,0,0,0,0,1,0,1,1], [3,1,2]))
# #zła kolejność bloków
# print(gradeFunc([0,1,1,1,0,0,1,1,0,0,1], [3,1,2]))
# #złe
# print(gradeFunc([0,1,0,1,0,0,0,1,0,1,1], [3,1,2]))
# print(gradeFunc([0,1,1,1,1,0,0,1,0,1,1], [3,1,2]))
# print(gradeFunc([0,1,1,1,1,0,0,0,0,1,1], [3,1,2]))
# print(gradeFunc([0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0], [3,1,2]))

def runAlgorythmV1(verticalPattern, horizontalPattern):
    hight = len(verticalPattern)
    width = len(horizontalPattern)

    def fitness_func(solution, solution_idx):
        rowsGrade = 0
        for i in range(hight):
            row = solution[i * width: i * width + width]
            grade = gradeFunc(row, horizontalPattern[i])
            rowsGrade += grade

        columnsGrade = 0
        for i in range(width):
            column = []
            for j in range(hight):
                column.append(solution[j * hight + i])
            grade = gradeFunc(column, verticalPattern[i])
            columnsGrade += grade

        fitness = rowsGrade + columnsGrade
        return fitness

    gene_space = [0, 1]
    num_genes = hight * width
    fitness_function = fitness_func
    sol_per_pop = 100

    num_parents_mating = 42
    num_generations = 500
    keep_parents = 6
    parent_selection_type = "sss"
    crossover_type = "single_point"
    mutation_type = "random"
    mutation_percent_genes = 1

    stop_criteria = "reach_0"

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

    genes = ga_instance.best_solution()[0]

    print("rows")
    for i in range(hight):
            row = genes[i * width: i * width + width]
            grade = gradeFunc(row, horizontalM[i])
            print(row, grade)

    print("columns")
    for i in range(width):
        column = []
        for j in range(hight):
            column.append(int(genes[j * hight + i]))
        grade = gradeFunc(column, verticalPattern[i])
        print(column, grade)

    print("Generations:", ga_instance.generations_completed, "Fitness:", ga_instance.best_solution()[1])


    img = createImg(genes, len(verticalM), len(horizontalM))

    plt.imshow(img, cmap="gray")
    ga_instance.plot_fitness()
    plt.show()
    return ga_instance

runAlgorythmV1(verticalM, horizontalM)