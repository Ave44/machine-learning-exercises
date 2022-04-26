from math import floor
import numpy as np
import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt
import time

verticalM = [[3,2],[1,3,2],[2,1,2,1],[2,1,1],[2,1],[1,2,1],[6,6],[11],[6,6],[1,1,1],[1,1,1],[1,1],[1,1,2,1],[1,1,2],[3]]
horizontalM = [[1,1],[1,1],[1,1],[1,1],[1,3],[3,3],[1,2,3,3],[1,1,2,1,1,1],[1,5,1,1],[2,3,1],[2,3,1],[1,3,1],[1,1,3,1,1],[2,5,1],[4,1,4]]

verticalM10 = [[1,1],[1,2,1],[2,3],[3,2,1],[4,3],[8,1],[3],[2,1],[1,1],[1]]
horizontalM10 = [[1],[2],[3],[4],[5],[1,2],[8],[6],[1,1,1,1,1],[1,1,1,1,1]]

def runAlgorythmSwarmV1(verticalPattern, horizontalPattern):     
    height = len(verticalPattern)
    width = len(horizontalPattern)
    x_max = np.ones(height*width, dtype=int)
    x_min = np.zeros(height*width, dtype=int)
    my_bounds = (x_min, x_max)
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

    def fitness_func(solution):
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

    def gradeFunc(solution, pattern):
        solutionPattern = [0]
        for i in solution:
            if round(i) == 0:
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

    def createImg(genes, height, width):
        img = np.zeros((height, width))
        index = 0
        for i in range(height):
            for j in range(width):
                img[i][j] = abs(genes[index]-1)
                index += 1
        
        return img

    def f(swarm):
        n_particles = swarm.shape[0]
        j = [-fitness_func(swarm[i]) for i in range(n_particles)]
        return np.array(j)

    optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=height*width, options=options, bounds=my_bounds)
    result = optimizer.optimize(f, iters=1000)

    pixels = result[1]
    for i in range(len(pixels)):
        pixels[i] = round(pixels[i])

    img = createImg(pixels, height, width)
    plt.imshow(img, cmap="gray")

    for i in range(height):
        for j in range(width):
            if img[i][j] == 0:
                print("██", end="")
            else:
                print("░░", end="")
        print("")

    plot_cost_history(optimizer.cost_history)
    plt.show()
    return result

def runAlgorythmSwarmV2(verticalPattern, horizontalPattern):
    def getNumberOfBlocks(pattern):
        num = 0
        for i in pattern:
            num += len(i)
        return num

    height = len(verticalPattern)
    width = len(horizontalPattern)
    blocks = getNumberOfBlocks(horizontalPattern)
    x_max = np.full((blocks), width, dtype=int)
    x_min = np.full((blocks), 0, dtype=int)
    my_bounds = (x_min, x_max)
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

    def gradeFuncV2(solution, pattern):
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
        grade = 0
        for i in range(len(pattern)):
            for j in range(i):
                if pattern[j][0] + pattern[j][1] >= pattern[i][0]:
                    grade -= 1000
                if pattern[j][0] == pattern[i][0]: # dodatkowe ujemne punkty żeby bloki na siebie nie nachodziły
                    grade -= 100000
        return grade

    def fitness_func(solution):
        fitness = 0
        img = np.ones((height, width), dtype=int)
        index = 0
        for i in range(len(horizontalPattern)):
            generatedPattern = []
            for j in horizontalPattern[i]:
                start = int(solution[index])
                generatedPattern.append([floor(solution[index]), j])
                if start + j <= width :
                    for n in range(j):
                        img[i][start + n] = 0
                else:
                    for n in range(width - start):
                        img[i][start + n] = 0
                    fitness -= abs(width - start - j)*10000
                index += 1
            fitness += gradeGenPatternV2(generatedPattern)
        columnMistakes = 0
        for i in range(width):
            column = []
            for j in range(height):
                column.append(img[j][i])
            columnMistakes += gradeFuncV2(column, verticalPattern[i])

        return columnMistakes + fitness

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

    def f(swarm):
        n_particles = swarm.shape[0]
        j = [-fitness_func(swarm[i]) for i in range(n_particles)]
        return np.array(j)

    optimizer = ps.single.GlobalBestPSO(n_particles=20000, dimensions=blocks, options=options, bounds=my_bounds)
    result = optimizer.optimize(f, iters=50)

    generatedPattern = result[1]
    for i in range(len(generatedPattern)):
        generatedPattern[i] = floor(generatedPattern[i])

    print("Block positions: ", generatedPattern)
    img = createImgV2(generatedPattern, height, width, horizontalPattern)
    plt.imshow(img, cmap="gray")

    for i in range(height):
        for j in range(width):
            if img[i][j] == 0:
                print("██", end="")
            else:
                print("░░", end="")
        print("")

    plot_cost_history(optimizer.cost_history)
    plt.show()
    return result

def measureSwarm(func, times, verticalPattern, horizontalPattern):
    timeArray = []
    fitnessArray = []

    for i in range(times):
        start = time.time()
        instance = func(verticalPattern, horizontalPattern)
        end = time.time()
        timeArray.append(end - start)
        fitnessArray.append(instance[0])
        print("===", i + 1, "/", times, "===")

    solved = 0
    for i in fitnessArray:
        if i == 0:
            solved += 1

    print("Time: ", timeArray, " Average: ", sum(timeArray)/times)
    print("Fitness: ", fitnessArray, " Average: ", sum(fitnessArray)/times, " Solved: ", solved)

# measureSwarm(runAlgorythmSwarmV2, 10, verticalM10, horizontalM10)
runAlgorythmSwarmV2(verticalM10, horizontalM10)