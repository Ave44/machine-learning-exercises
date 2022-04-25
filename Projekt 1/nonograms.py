import pygad
import numpy as np
import matplotlib.pyplot as plt
import time

# motyl
verticalM = [[3,2],[1,3,2],[2,1,2,1],[2,1,1],[2,1],[1,2,1],[6,6],[11],[6,6],[1,1,1],[1,1,1],[1,1],[1,1,2,1],[1,1,2],[3]]
horizontalM = [[1,1],[1,1],[1,1],[1,1],[1,3],[3,3],[1,2,3,3],[1,1,2,1,1,1],[1,5,1,1],[2,3,1],[2,3,1],[1,3,1],[1,1,3,1,1],[2,5,1],[4,1,4]]

# statek
verticalM10 = [[1,1],[1,2,1],[2,3],[3,2,1],[4,3],[8,1],[3],[2,1],[1,1],[1]]
horizontalM10 = [[1],[2],[3],[4],[5],[1,2],[8],[6],[1,1,1,1,1],[1,1,1,1,1]]

# kaczka
verticalM102 = [[1],[2,3],[4,1,1],[1,3],[1,1,1,2],[1,1,2],[4,1,1],[1,1],[1,1],[5]]
horizontalM102 = [[3],[1,1],[1,1,1],[3,1],[5,1],[1,1,1],[1,4],[1,1,1],[1,2,2],[2,3,1]]

# papuga
verticalM15 = [[5,2],[2,1,1,2],[1,3,2],[1,1,2,2],[1,2,2,2],[2,1,4,1],[5,1,2],[2,2,4],[2,1,1,2],[1,2,2,2],[2,2,1],[2,7],[2,2,2],[2,2,2],[5,1]]
horizontalM15 = [[5],[2,2],[1,1,3],[3,5],[1,1,2,2],[4,2,2],[1,3,2],[2,2,2],[2,2,1],[2,2,1],[2,3,4],[4,1,1,1,2],[5,3,1],[6,3],[6]]

# flamingi
verticalM20 = [[1,2],[2,2,4],[3,3,1,5],[1,8,1,5],[8,2,2,3],[4,1,1,2],[1,1,1,1],[2,2,2],[4,1],[5,1],[6],[2,1,2],[2,1],[5,1],[2,1],[1,2,2,1],[2,1,2,1],[5,1,2],[2,1,1],[2,1,1]]
horizontalM20 = [[3],[2,1],[5],[2,1,2],[2,2,3],[3,3,1],[9,3],[4,4,1,3],[2,2,3,1],[2,3,4],[4],[1,2],[4,3,3],[4,1,1],[4],[3,1,3],[5,1,3],[6,6],[4],[3]]

# tważ
verticalM30x40 = [[3,2],[2,3,1,1],[1,3,1,2,1],[3,1,1,2,2,5],[2,2,1,2,1,9],[2,1,2,2,6,4],[2,2,2,1,1,7,2],[1,3,2,2,9,1],[1,3,2,1,1,10],[1,2,2,1,1,1,4,6],[1,2,2,3,3,5,5],[1,2,2,2,1,1,2,7,4],[1,3,3,2,1,2,9,4],[2,2,5,3,12,3],[2,3,9,3,8,2],[1,2,19,6,2],[2,3,18,5,1],[1,3,9,6,5,1],[2,4,6,5,4,1],[2,6,4,4,4],[5,10,4,3,3],[12,4,3,3,3],[2,10,3,3,3,3],[2,2,9,3,2,3,4],[1,2,6,3,3,3,2,1],[3,1,5,3,2,3,2,1],[1,2,1,4,3,1,3,2,1],[2,1,1,2,2,5,2,2,1],[4,1,2,3,3,5,3,2,2],[4,1,2,6,2,2,3,2,2]]
horizontalM30x40 = [[3,10,2,1,2],[2,4,4,2,1,3],[1,3,4,3,1,1,3],[3,9,2,2,2],[3,5,5,2,1,2],[2,2,4,4,1,2,2],[4,3,1,2],[4,2,2,3],[3,1,2,2],[5,3,2,1],[4,3,1,2],[2,6,4],[1,1,3,3,5],[4,1,3,5],[2,3,1,6,5,2],[1,5,4,4],[2,3,4,2],[2,1,3,5,1],[1,3,5,1],[4,1,4,2],[4,4,2,7],[2,2,5,1,5],[3,6,2,3,1],[4,2,3,5],[1,6,3,2],[2,2,3,2,7],[3,3,4,2,6],[9,4,3,4],[11,4,4],[11,4,4],[12,4,3,1],[6,6,5,2],[2,4,6,9],[2,4,6,7],[3,4,6,3,1],[2,5,6,3],[3,5,14],[2,5,10,1],[1,6,5,2],[7,6]]

# ptak
verticalMColor = [[[1,1]],[[1,1],[2,1]],[[2,1],[4,1]],[[1,2],[6,1]],[[3,2],[4,1]],[[4,2],[2,1],[1,3]],[[3,2],[2,1],[3,3]],[[1,1],[2,2],[1,1],[3,3]],[[2,1],[1,2],[1,1],[3,3],[2,1]],[[5,1],[5,3],[1,1]],[[4,1],[5,3]],[[2,1],[1,1],[4,3]],[[3,1],[2,3]],[[1,1]],[[1,1]]]
horizontalMColor = [[[3,1]],[[5,1]],[[4,1],[3,1]],[[4,2],[4,1]],[[4,2],[1,1],[3,3]],[[3,2],[1,1],[4,3]],[[3,2],[1,1],[3,3]],[[6,1],[4,3]],[[5,1],[5,3]],[[2,1],[5,3]],[[3,1],[2,3],[1,1]],[[3,1],[2,1]],[[4,1]],[[1,1]]]
colors = {1: [0, 0, 0], 2: [110, 110, 110], 3: [176, 30, 30]}

# funkcja zwraca najniejszą możliwą wartość tak żeby przynajmniej jeden gen został wybrany
def getPercentageOfMutations(num_genes):
    return 100/num_genes

def runAlgorythmV1(verticalPattern, horizontalPattern):
    height = len(verticalPattern)
    width = len(horizontalPattern)

    def createImg(genes, height, width):
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

        if len(filteredRowPatern) > len(pattern):
            for i in range(len(filteredRowPatern) - len(pattern)):
                grade -= filteredRowPatern[i + len(pattern)]

        return grade

    def fitness_func(solution, solution_idx):
        rowsGrade = 0
        for i in range(height):
            row = solution[i * width: i * width + width]
            grade = gradeFunc(row, horizontalPattern[i])
            rowsGrade += grade

        columnsGrade = 0
        for i in range(width):
            column = []
            for j in range(height):
                column.append(solution[j * height + i])
            grade = gradeFunc(column, verticalPattern[i])
            columnsGrade += grade

        fitness = rowsGrade + columnsGrade
        return fitness

    gene_space = [0, 1]
    num_genes = height * width
    fitness_function = fitness_func
    sol_per_pop = 100

    num_parents_mating = 42
    num_generations = 5000
    keep_parents = 6
    parent_selection_type = "sss"
    crossover_type = "single_point"
    mutation_type = "random"
    mutation_percent_genes = getPercentageOfMutations(num_genes)
    stop_criteria = ["reach_0", "saturate_200"]

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

    if ga_instance.best_solution()[1] != 0:
        print("Mistakes in paterns:")

        print("rows")
        for i in range(height):
                row = genes[i * width: i * width + width]
                grade = gradeFunc(row, horizontalPattern[i])
                print(row, grade)

        print("columns")
        for i in range(width):
            column = []
            for j in range(height):
                column.append(int(genes[j * height + i]))
            grade = gradeFunc(column, verticalPattern[i])
            print(column, grade)

    img = createImg(genes, height, width)

    # wyświetlanie wygenerowanego obrazka w konsoli
    for i in range(height):
            for j in range(width):
                if img[i][j] == 0:
                    print("██", end="")
                else:
                    print("░░", end="")
            print("")

    plt.imshow(img, cmap="gray")
    ga_instance.plot_fitness()
    plt.show()
    return ga_instance

def runAlgorythmV2(verticalPattern, horizontalPattern):
    height = len(horizontalPattern)
    width = len(verticalPattern)

    def createImgV2(genes, height, width, horizontalPattern):
        img = np.ones((height, width), dtype=int)
        index = 0
        for i in range(len(horizontalPattern)):
            for j in horizontalPattern[i]:
                start = int(genes[index])
                if start + j <= width :
                    for n in range(j):
                        img[i][start + n] = 0
                else:
                    for n in range(width - start):
                        img[i][start + n] = 0
                index += 1
        
        return img

    def gradeFuncV2(solution, pattern):
        # tworzenie "wzoru" z wygenerowanych wartości
        solutionPattern = [-1]
        for i in solution:
            if i == 1:
                solutionPattern.append(-1)
            else:
                if solutionPattern[-1] != -1:
                    solutionPattern[-1] = solutionPattern[-1] + 1
                else:
                    solutionPattern.append(1)
        filteredRowPatern = list(filter(lambda n: n != -1 , solutionPattern))
        # ocenianie wygenerowaniego "wzoru"
        grade = -abs(sum(pattern) - sum(filteredRowPatern))

        for i in range(len(pattern)):
            if i < len(filteredRowPatern):
                if pattern[i] != filteredRowPatern[i]:
                    grade -= abs(pattern[i] - filteredRowPatern[i])
            else:
                grade -= pattern[i]

        if len(filteredRowPatern) > len(pattern):
                for i in range(len(filteredRowPatern) - len(pattern)):
                    grade -= filteredRowPatern[i + len(pattern)]

        return grade

    def gradeGenPatternV2(pattern):
        # pattern[i][0] początek bloku
        # pattern[i][0] + pattern[i][1] ostatni index zajmowany przez ten blok (długość bloku + 1 miejsce przerwy)
        grade = 0
        for i in range(len(pattern)):
            # prównywanie obecnego bloku do wszystkich poprzednich
            for j in range(i):
                if pattern[j][0] + pattern[j][1] >= pattern[i][0]:
                    grade -= 1000
        return grade

    def fitness_func(solution, solution_idx):
        fitness = 0
        # tworzenie macieży
        img = np.ones((height, width), dtype=int)
        index = 0
        for i in range(len(horizontalPattern)):
            generatedPattern = []
            for j in horizontalPattern[i]:
                start = int(solution[index])
                generatedPattern.append([solution[index], j])
                if start + j <= width :
                    for n in range(j):
                        img[i][start + n] = 0
                else:
                    for n in range(width - start):
                        img[i][start + n] = 0
                    # odejmuj punkty gdy blok wychodzi poza wiersz
                    fitness -= abs(width - start - j)*10000
                index += 1
            fitness -= abs(gradeGenPatternV2(generatedPattern))
        # ocenianie kolumn
        for i in range(width):
            column = []
            for j in range(height):
                column.append(img[j][i])
            fitness += gradeFuncV2(column, verticalPattern[i])

        return fitness

    def getNumberOfBlocks(pattern):
        num = 0
        for i in pattern:
            num += len(i)
        return num

    gene_space = list(range(0, width))
    num_genes = getNumberOfBlocks(horizontalPattern)
    fitness_function = fitness_func
    sol_per_pop = 200

    num_parents_mating = 99
    num_generations = 5000
    keep_parents = 2
    parent_selection_type = "sss"
    # crossover_type = "single_point" # two points działa znacznie lepiej
    crossover_type = "two_points"
    mutation_type = "random"
    mutation_percent_genes = getPercentageOfMutations(num_genes) # im mniejsza wartość tym lepiej

    stop_criteria = ["reach_0", "saturate_200"]

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

    img = createImgV2(genes, height, width, horizontalPattern)
    plt.imshow(img, cmap="gray")

    if ga_instance.best_solution()[1] != 0:
        print("Mistakes in vertical paterns:")
        for i in range(width):
            column = []
            for j in range(height):
                column.append(img[j][i])
            print(column, gradeFuncV2(column, verticalPattern[i]))

    for i in range(height):
        for j in range(width):
            if img[i][j] == 0:
                print("██", end="")
            else:
                print("░░", end="")
        print("")

    ga_instance.plot_fitness()
    plt.show()
    return ga_instance

def runAlgorythmColorV2(verticalPattern, horizontalPattern, colors):
    height = len(horizontalPattern)
    width = len(verticalPattern)

    def gradeGenPatternColor(pattern):
        # pattern[i][0] początek bloku
        # pattern[i][0] + pattern[i][1] ostatni index zajmowany przez ten blok (długość bloku + 1 miejsce przerwy)
        grade = 0
        for i in range(len(pattern)):
            # prównywanie obecnego bloku do wszystkich poprzednich
            for j in range(i):
                if pattern[j][2] != pattern[i][2]:
                    if pattern[j][0] + pattern[j][1] - 1 >= pattern[i][0]: # odejmuje 1 żeby pozbyć się przerwy między blokami
                        grade -= 1000
                else:
                    if pattern[j][0] + pattern[j][1] >= pattern[i][0]:
                        grade -= 1000
        return grade

    def gradePaternsByColorLength(genPattern, pattern):
        # funkcja porównuje ile pixeli danego koloru jest w każdym wzorze
        grade = 0
        original = {}
        generated = {}
        for i in pattern: # i jest postaci [długość bloku, kolor]
            if i[1] in original:
                original[i[1]] = original[i[1]] + i[0]
            else:
                original[i[1]] = i[0]
        for i in genPattern:
            if i[1] in generated:
                generated[i[1]] = generated[i[1]] + i[0]
            else:
                generated[i[1]] = i[0]

        for i in original:
            if i in generated:
                grade -= abs(original[i] - generated[i])
                del generated[i]
            else:
                grade -= original[i]
        for i in generated:
            if i in original:
                grade -= abs(original[i] - generated[i])
            else:
                grade -= generated[i]

        return grade

    def gradeFuncColor(solution, pattern):
        # tworzenie "wzoru" z wygenerowanych wartości
        solutionPattern = [-1]
        for i in solution:
            if i == 0:
                solutionPattern.append(-1)
            else:
                if solutionPattern[-1] != -1:
                    solutionPattern[-1][0] = solutionPattern[-1][0] + 1
                else:
                    solutionPattern.append([1,i])
        filteredRowPatern = list(filter(lambda n: n != -1 , solutionPattern))
        # ocenianie wygenerowaniego "wzoru"
        grade = gradePaternsByColorLength(filteredRowPatern, pattern)

        for i in range(len(pattern)):
            if i < len(filteredRowPatern):
                if pattern[i] != filteredRowPatern[i]:
                    grade -= abs(pattern[i][0] - filteredRowPatern[i][0])
                    if pattern[i][1] != filteredRowPatern[i][1]: # wtedy gdy kolory są różne
                        grade -= pattern[i][0]
            else:
                grade -= pattern[i][0]

        if len(filteredRowPatern) > len(pattern):
            for i in range(len(filteredRowPatern) - len(pattern)):
                grade -= filteredRowPatern[i + len(pattern)][0]

        return grade

    def createImgColor(genes, height, width, horizontalPattern):
        img = np.full((height, width, 3), 255, dtype=int)
        index = 0
        for i in range(len(horizontalPattern)):
            for j in horizontalPattern[i]:
                start = int(genes[index])
                if start + j[0] <= width :
                    for n in range(j[0]):
                        img[i][start + n] = colors[j[1]]
                else:
                    for n in range(width - start):
                        img[i][start + n] = colors[j[1]]
                index += 1
        
        return img

    def fitness_func(solution, solution_idx):
        fitness = 0
        # tworzenie macieży
        img = np.zeros((height, width), dtype=int)
        index = 0
        for i in range(len(horizontalPattern)):
            generatedPattern = []
            for j in horizontalPattern[i]:
                start = int(solution[index])
                generatedPattern.append([solution[index], j[0], j[1]]) # [początek bloku, długość bloku, kolor bloku]
                if start + j[0] <= width : # sprawdzenie czy blok nie wychodzi poza obraz
                    for n in range(j[0]):
                        img[i][start + n] = j[1]
                else:
                    for n in range(width - start):
                        img[i][start + n] = j[1]
                    # odejmuj punkty gdy blok wychodzi poza wiersz
                    fitness -= abs(width - start - j[0])*10000
                index += 1
            fitness -= abs(gradeGenPatternColor(generatedPattern))
        # ocenianie kolumn
        for i in range(width):
            column = []
            for j in range(height):
                column.append(img[j][i])
            fitness += gradeFuncColor(column, verticalPattern[i])

        return fitness

    def getNumberOfBlocks(pattern):
        num = 0
        for i in pattern:
            num += len(i)
        return num

    gene_space = list(range(0, width))
    num_genes = getNumberOfBlocks(horizontalPattern)
    fitness_function = fitness_func
    sol_per_pop = 200

    num_parents_mating = 99
    num_generations = 5000
    keep_parents = 2
    parent_selection_type = "sss"
    crossover_type = "single_point"
    mutation_type = "random"
    mutation_percent_genes = getPercentageOfMutations(num_genes) # im mniejsza wartość tym lepiej

    stop_criteria = ["reach_0", "saturate_200"]

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

    img = createImgColor(genes, height, width, horizontalPattern)
    plt.imshow(img)

    if ga_instance.best_solution()[1] != 0:
        print("Mistakes in vertical paterns:")
        img = np.zeros((height, width), dtype=int)
        index = 0
        for i in range(len(horizontalPattern)):
            for j in horizontalPattern[i]:
                start = int(genes[index])
                if start + j[0] <= width :
                    for n in range(j[0]):
                        img[i][start + n] = j[1]
                else:
                    for n in range(width - start):
                        img[i][start + n] = j[1]
                index += 1
        for i in range(width):
            column = []
            for j in range(height):
                column.append(img[j][i])
            print(column, verticalPattern[i], gradeFuncColor(column, verticalPattern[i]))

    ga_instance.plot_fitness()
    plt.show()
    return ga_instance

def runAlgorythmColorV1(verticalPattern, horizontalPattern, colors):
    height = len(horizontalPattern)
    width = len(verticalPattern)

    def createImgColor(genes, height, width):
        img = np.full((height, width, 3), 255, dtype=int)
        index = 0
        for i in range(height):
            for j in range(width):
                if genes[index] != 0:
                    img[i][j] = colors[genes[index]]
                index += 1
        
        return img

    def gradePaternsByColorLength(genPattern, pattern):
        # funkcja porównuje ile pixeli danego koloru jest w każdym wzorze
        grade = 0
        original = {}
        generated = {}
        for i in pattern: # i jest postaci [długość bloku, kolor]
            if i[1] in original:
                original[i[1]] = original[i[1]] + i[0]
            else:
                original[i[1]] = i[0]
        for i in genPattern:
            if i[1] in generated:
                generated[i[1]] = generated[i[1]] + i[0]
            else:
                generated[i[1]] = i[0]

        for i in original:
            if i in generated:
                grade -= abs(original[i] - generated[i])
                del generated[i]
            else:
                grade -= original[i]
        for i in generated:
            if i in original:
                grade -= abs(original[i] - generated[i])
            else:
                grade -= generated[i]

        return grade
        
    def gradeFuncColor(solution, pattern):
        # tworzenie "wzoru" z wygenerowanych wartości
        solutionPattern = [-1]
        for i in solution:
            if i == 0:
                solutionPattern.append(-1)
            else:
                if solutionPattern[-1] != -1:
                    solutionPattern[-1][0] = solutionPattern[-1][0] + 1
                else:
                    solutionPattern.append([1,i])
        filteredRowPatern = list(filter(lambda n: n != -1 , solutionPattern))
        # ocenianie wygenerowaniego "wzoru"
        grade = gradePaternsByColorLength(filteredRowPatern, pattern)*10

        for i in range(len(pattern)):
            if i < len(filteredRowPatern):
                if pattern[i] != filteredRowPatern[i]:
                    grade -= abs(pattern[i][0] - filteredRowPatern[i][0])
                    if pattern[i][1] != filteredRowPatern[i][1]: # wtedy gdy kolory są różne
                        grade -= pattern[i][0]
            else:
                grade -= pattern[i][0]

        if len(filteredRowPatern) > len(pattern):
            for i in range(len(filteredRowPatern) - len(pattern)):
                grade -= filteredRowPatern[i + len(pattern)][0]

        return grade

    def fitness_func(solution, solution_idx):
        rowsGrade = 0
        for i in range(height):
            row = solution[i * width: i * width + width]
            grade = gradeFuncColor(row, horizontalPattern[i])
            rowsGrade += grade

        columnsGrade = 0
        for i in range(width):
            
            column = []
            for j in range(height):
                column.append(solution[j * height + i])
            grade = gradeFuncColor(column, verticalPattern[i])
            columnsGrade += grade

        fitness = rowsGrade + columnsGrade
        return fitness
    
    gene_space = list(colors.keys()) + [0]
    num_genes = height * width
    fitness_function = fitness_func
    sol_per_pop = 100

    num_parents_mating = 42
    num_generations = 5000
    keep_parents = 6
    parent_selection_type = "sss"
    crossover_type = "single_point"
    # crossover_type = "two_points" # mniej efektywne
    mutation_type = "random"
    mutation_percent_genes = getPercentageOfMutations(num_genes)
    stop_criteria = ["reach_0", "saturate_200"]

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

    if ga_instance.best_solution()[1] != 0:
        print("Mistakes in paterns:")

        print("rows")
        for i in range(height):
            row = genes[i * width: i * width + width]
            grade = gradeFuncColor(row, horizontalPattern[i])
            print(row, grade)

        print("columns")
        for i in range(width):
            column = []
            for j in range(height):
                column.append(int(genes[j * height + i]))
            grade = gradeFuncColor(column, verticalPattern[i])
            print(column, grade)

    img = createImgColor(genes, height, width)

    plt.imshow(img, cmap="gray")
    ga_instance.plot_fitness()
    plt.show()
    return ga_instance

# żeby użyć tej funkcji należy zakomentować część z wyświetlaniem obrazów i wykresów
def measure(func, times, verticalPattern, horizontalPattern):
    timeArray = []
    fitnessArray = []
    generationsArray = []

    for i in range(times):
        start = time.time()
        instance = func(verticalPattern, horizontalPattern)
        end = time.time()
        timeArray.append(end - start)
        fitnessArray.append(instance.best_solution()[1])
        generationsArray.append(instance.generations_completed)
        print("===", i + 1, "/", times, "===")

    solved = 0
    for i in fitnessArray:
        if i == 0:
            solved += 1

    print("Time: ", timeArray, " Average: ", sum(timeArray)/times)
    print("Fitness: ", fitnessArray, " Average: ", sum(fitnessArray)/times, " Solved: ", solved)
    print("Generations: ", generationsArray, " Average: ", sum(generationsArray)/times)

measure(runAlgorythmV2, 100, verticalM, horizontalM)

# runAlgorythmV1(verticalM10, horizontalM10)
# runAlgorythmColorV1(verticalMColor, horizontalMColor, colors )