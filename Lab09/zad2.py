import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d

# A)

# tworzymy tablice o wymiarach 128x128x3 (3 kanaly to RGB)
# uzupelnioną zerami = kolor czarny
data = np.zeros((128, 128, 3), dtype=np.uint8)


# chcemy zeby obrazek byl czarnobialy,
# wiec wszystkie trzy kanaly rgb uzupelniamy tymi samymi liczbami
# napiszmy do tego funkcje
def draw(img, x, y, color):
    img[x, y] = [color, color, color]


# zamalowanie 4 pikseli w lewym górnym rogu
draw(data, 5, 5, 100)
draw(data, 6, 6, 100)
draw(data, 5, 6, 255)
draw(data, 6, 5, 255)


# rysowanie kilku figur na obrazku
for i in range(128):
    for j in range(128):
        if (i-64)**2 + (j-64)**2 < 900:
            draw(data, i, j, 200)
        elif i > 100 and j > 100:
            draw(data, i, j, 255)
        elif (i-15)**2 + (j-110)**2 < 25:
            draw(data, i, j, 150)
        elif (i-15)**2 + (j-110)**2 == 25 or (i-15)**2 + (j-110)**2 == 26:
            draw(data, i, j, 255)

# konwersja macierzy na obrazek i wyświetlenie
# plt.imshow(data, interpolation='nearest')

# B)

R = data[:,:,0]
G = data[:,:,1]
B = data[:,:,2]

horizontalFilter = [[1,1,1],[0,0,0],[-1,-1,-1]]
filteredR = convolve2d(R, horizontalFilter)
filteredG = convolve2d(G, horizontalFilter)
filteredB = convolve2d(B, horizontalFilter)

filtered = np.empty((130,130,3))
filtered[:,:,0] = filteredR
filtered[:,:,1] = filteredG
filtered[:,:,2] = filteredB

# plt.figure(figsize=(8,4))
# plt.subplot(1,2,1)
# plt.imshow(data, cmap="gray")
# plt.title("Original image")

# plt.subplot(1,2,2)
# plt.imshow(filtered, cmap="gray")
# plt.title("Filtered image")

# C)

def reLU(x):
    if x < 0:
        return 0
    return x

def aplayFuncToRGB(img, func):
    resImg = img
    height = len(img)
    width = len(img[0])
    for i in range(height):
        for j in range(width):
            resImg[i][j][0] = func(resImg[i][j][0])
            resImg[i][j][1] = func(resImg[i][j][1])
            resImg[i][j][2] = func(resImg[i][j][2])
    return resImg

preprocesed = aplayFuncToRGB(filtered, reLU)

# plt.figure(figsize=(8,4))
# plt.subplot(1,2,1)
# plt.imshow(filtered, cmap="gray")
# plt.title("Filtered image")

# plt.subplot(1,2,2)
# plt.imshow(preprocesed, cmap="gray")
# plt.title("Preprocesed image")

# D)

def reLUAdvanced(x):
    if x < 0:
        return 0
    if x <= 255:
        return x
    return 255

preprocesed2 = aplayFuncToRGB(filtered, reLUAdvanced)

# plt.figure(figsize=(8,4))
# plt.subplot(1,2,1)
# plt.imshow(filtered, cmap="gray")
# plt.title("Filtered image")

# plt.subplot(1,2,2)
# plt.imshow(preprocesed2, cmap="gray")
# plt.title("Preprocesed image 2")

plt.figure(figsize=(8,8))
plt.suptitle('Horizontal filter')
plt.subplot(2,2,1)
plt.imshow(data, cmap="gray")
plt.title("Original image")

plt.subplot(2,2,2)
plt.imshow(filtered, cmap="gray")
plt.title("Filtered image")

plt.subplot(2,2,3)
plt.imshow(preprocesed, cmap="gray")
plt.title("Preprocesed image")

plt.subplot(2,2,4)
plt.imshow(preprocesed2, cmap="gray")
plt.title("Preprocesed image 2")

# E)

def testKernel(kernel, title):
    R = data[:,:,0]
    G = data[:,:,1]
    B = data[:,:,2]

    filteredR = convolve2d(R, kernel)
    filteredG = convolve2d(G, kernel)
    filteredB = convolve2d(B, kernel)

    filtered = np.empty((130,130,3))
    filtered[:,:,0] = filteredR
    filtered[:,:,1] = filteredG
    filtered[:,:,2] = filteredB

    preprocesed = aplayFuncToRGB(filtered, reLU)
    preprocesed2 = aplayFuncToRGB(filtered, reLUAdvanced)

    plt.figure(figsize=(8,8))
    plt.suptitle(title)
    plt.subplot(2,2,1)
    plt.imshow(data, cmap="gray")
    plt.title("Original image")

    plt.subplot(2,2,2)
    plt.imshow(filtered, cmap="gray")
    plt.title("Filtered image")

    plt.subplot(2,2,3)
    plt.imshow(preprocesed, cmap="gray")
    plt.title("Preprocesed image")

    plt.subplot(2,2,4)
    plt.imshow(preprocesed2, cmap="gray")
    plt.title("Preprocesed image 2")

verticalFilter = [[ 1, 0,-1],
                  [ 1, 0,-1],
                  [ 1, 0,-1]]
testKernel(verticalFilter, 'Vertical filter')

# F)

sobel45 = [[ 0, 1, 2],
           [-1, 0, 1],
           [-2,-1, 0]]
testKernel(sobel45, 'Sobel filter 45°')

sobel135 = [[ 2, 1, 0],
            [ 1, 0,-1],
            [ 0,-1,-2]]
testKernel(sobel135, 'Sobel filter 135°')

plt.show()