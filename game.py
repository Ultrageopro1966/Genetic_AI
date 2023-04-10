import pygame, time, os
import numpy as np
from AI import AI
import random
from matplotlib import pyplot as plt

pygame.init()
WIDTH, HEIGHT = 1000, 500 # Размеры экрана
speed = 2 # Скорость спрайтов
sampleSize = 150 # Размер выборки
MAIN_AI = AI((WIDTH, HEIGHT), 1.8) # Подключение искусственного интеллекта (rate - сила мутации)

# Подключение/создание необходимых переменных pygame и критериев остановки
BALLS = pygame.sprite.Group()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
start, frame, count, clear = 0, 0, 0, False
f1 = pygame.font.SysFont('Arial', 48)
running = True
image = pygame.transform.scale(pygame.image.load("bacteria.png"), (50, 50))
text1, text2 = f1.render(f"Итерация {count}", False, (100, 100, 100)), f1.render(f"Скорость - {speed}", False, (100, 100, 200))

# Переменные для графика
x_data = 0
y_data = []

# Создание класса спрайта
class ball(pygame.sprite.Sprite):
    def __init__(self, speed:float) -> None:
        super().__init__()
        # Инициализация атрибутов спрайта
        self.image = image #Загрузка изображения
        self.rect = self.image.get_rect()
        self.rect.center = (WIDTH/2, HEIGHT/2)
        self.weights = MAIN_AI.mutation() #Мутацитя весов
        self.speed = speed
        self.positions = []
    def update(self) -> None:
        global clear, start, frame, x_data
        coordinates = np.array([self.rect.x, self.rect.y]) # Вектор координат спрайта
        move = MAIN_AI.predict(coordinates, *self.weights) # Направление движения спрайта, предсказанное AI
        timeControl = (time.time_ns() - start)/1000000000
        #Контроль зацикленных спрайтов
        if frame%30 == 0:
            self.positions.append(coordinates)
        if len(self.positions) >= 2:
            if np.sum((self.positions[-1] - self.positions[-2]) ** 2) ** 0.5 < 20:
                self.kill()
        
        if self.rect.x < 0 or self.rect.x > 950\
        or self.rect.y > 450 or self.rect.y < 0: #Контроль перехода грниц
            self.kill()
        
        if timeControl > 15: #Контроль успешности модели по времени (без остановки)
            MAIN_AI.W1, MAIN_AI.W2 = self.weights
        
        if len(BALLS)<3: #Контроль успешности модели по остатку (с остановкой)
            clear = True
            try: #Попытка скрещивания весов
                MAIN_AI.W1 = np.array((BALLS.sprites()[0].weights[0] + BALLS.sprites()[1].weights[0])/2)
                MAIN_AI.W2 = np.array((BALLS.sprites()[0].weights[1] + BALLS.sprites()[1].weights[1])/2)
            except:
                MAIN_AI.W1, MAIN_AI.W2 = self.weights
            
            if timeControl < 100000000:
                x_data+=1
                y_data.append(timeControl)

        match move: #Движение спрайта
            case 0:
                self.rect.x -= self.speed
            case 1:
                self.rect.x += self.speed
            case 2:
                self.rect.y -= self.speed
            case _:
                self.rect.y += self.speed

for _ in range(sampleSize): #Генерация первой выборки
    BALLS.add(ball(speed))

#Игровой цикл
while running:
    screen.fill((255, 255, 255))
    BALLS.draw(screen)
    screen.blit(text1, (10, 50))
    screen.blit(text2, (10, 100))
    BALLS.update()
    frame+=1
    if clear: #Обновление выборки
        BALLS.empty()
        clear, frame = False, 0
        start = time.time_ns()
        count+=1
        text1, text2 = f1.render(f"Итерация {count}", False, (100, 100, 100)), f1.render(f"Скорость - {speed}", False, (100, 100, 200))
        for _ in range(sampleSize):
            BALLS.add(ball(speed))
        os.system("cls")
        print(MAIN_AI.W1,"\n\n", MAIN_AI.W2)
    pygame.display.flip()
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

plt.plot(range(x_data), y_data)
plt.xlabel("Итерация")
plt.ylabel("Время жизни")
plt.show()