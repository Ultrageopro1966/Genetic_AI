import numpy as np
import random
from matplotlib import pyplot as plt

class AI:
    def __init__(self, size:tuple, rate:float) -> None:
        self.width = size[0]
        self.height = size[1]
        self.rate = rate #Коэффициент значимости мутации
        self.W1 = np.random.random((5, 4)) #Веса первого слоя (матрица 5x4)
        self.W2 = np.random.random((4, 5)) #Веса первого слоя (матрица 4x5)
    
    def ReLU(self, x) -> float: #Функция активации скрытого слоя
        return max(0, x)

    def softmax(self, values:np.ndarray) -> np.ndarray: #Функция активации второго слоя 
        exponentValues = np.power(np.e, values)
        return exponentValues/sum(exponentValues)

    def normalize(self, values:np.ndarray) -> np.ndarray: #Стандартизация и нормализация координат
        xMeanings = np.array([values[0], self.width - values[0]])/self.width
        yMeanings = np.array([values[1], self.height - values[1]])/self.height
        return np.concatenate((xMeanings, yMeanings))

    def predict(self, inputs:np.ndarray, w1:np.ndarray, w2:np.ndarray) -> int: #Предсказание сети 
        inputs = self.normalize(inputs)
        firstOutput = np.vectorize(self.ReLU)(w1 @ inputs.T)
        endOutput = self.softmax(w2 @ firstOutput)
        return np.argmax(endOutput)

    def mutation(self) -> np.ndarray: #Мутация весов
        return np.array([self.W1, self.W2]) if random.randint(1, 30) == 1 else np.array([self.W1 + np.random.random(self.W1.shape) * random.choice([-1, 1]) * self.rate,
                                                                                        self.W2 + np.random.random(self.W2.shape) * random.choice([-1, 1]) * self.rate]) 


